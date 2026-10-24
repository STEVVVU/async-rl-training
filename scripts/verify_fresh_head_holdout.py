"""Freeze and evaluate selected digit heads on a disjoint wider-operand holdout."""

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SELECTION_SHA256 = "57761fb394ceef4ffaa1eeebf0418952c308d6bc325f6a20642f99222e559beb"
PAIR_SEED = 20261024
PAIR_COUNT = 256
ORACLE_PAIR_COUNT = 16
MAX_EVALUATION_SECONDS = 600
MIN_FREE_DISK_BYTES = 10 * 1024**3
CPU_THREADS = 1


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fresh_pairs(seed=PAIR_SEED, count=PAIR_COUNT):
    pairs = [(a, b) for a in range(10, 100) for b in range(a, 100)]
    random.Random(seed).shuffle(pairs)
    selected = pairs[:count]
    if len(selected) != count or len(set(selected)) != count or any(a > b for a, b in selected):
        raise ValueError("Fresh pair protocol is incomplete or duplicated")
    return [list(pair) for pair in selected]


def protocol():
    return {
        "operand_min": 10,
        "operand_max": 99,
        "unordered_pairs": True,
        "pair_count": PAIR_COUNT,
        "shuffle_seed": PAIR_SEED,
        "pairs": fresh_pairs(),
        "oracle_pair_indices": list(range(ORACLE_PAIR_COUNT)),
        "oracle_rule": "first 16 frozen protocol pairs, tested for initial and all 10 selected heads",
        "tool_reward": "existing _prompts/_evaluate_study behavior, unchanged",
    }


def _check_old_receipts(selection_dir, model_path):
    selection_dir, model_path = Path(selection_dir).resolve(), Path(model_path).resolve()
    selection_path = selection_dir / "selection.json"
    if sha(selection_path) != SELECTION_SHA256:
        raise ValueError("Existing frozen selection does not have its predeclared SHA256")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_accessed") or selection.get("training_sha256") != sha(selection_dir / "training.json"):
        raise ValueError("Existing training/selection receipt identity mismatch")
    if not selection_dir.samefile(Path(selection["input_paths"]["saved_initial_head"]).resolve().parent):
        raise ValueError("Selection and initial-head artifact must share the original output directory")
    for name, digest in selection["inputs"].items():
        if sha(selection["input_paths"][name]) != digest:
            raise ValueError(f"Existing selection input hash mismatch: {name}")
    old_sources = {
        "train.py": ROOT / "examples/local_async/train.py",
        "driver.py": ROOT / "scripts/verify_head_quality.py",
    }
    for name, digest in selection["source"].items():
        if name not in old_sources or sha(old_sources[name]) != digest:
            raise ValueError(f"Existing selection source hash mismatch: {name}")
    training = json.loads((selection_dir / "training.json").read_text(encoding="utf-8"))
    for candidate in selection["selections"].values():
        if candidate["checkpoint"] and sha(selection_dir / candidate["checkpoint"]) != candidate["sha256"]:
            raise ValueError("Selected checkpoint hash mismatch")
    for run in training["runs"].values():
        for update in run["updates"]:
            if "checkpoint" in update and sha(selection_dir / update["checkpoint"]) != update["sha256"]:
                raise ValueError("Candidate checkpoint hash mismatch")
    model_hashes = {
        path.relative_to(model_path).as_posix(): sha(path) for path in sorted(model_path.rglob("*")) if path.is_file()
    }
    if model_hashes != training["manifest"]["prior_model_sha256"]:
        raise ValueError("Cached pretrained model assets differ from frozen training")
    return selection, training, model_hashes


def freeze(selection_dir, model_path, output):
    output = Path(output)
    selection, training, model_hashes = _check_old_receipts(selection_dir, model_path)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": 1,
        "claim_scope": "wider-operand arithmetic distribution-shift head evaluation",
        "selection_dir": str(Path(selection_dir).resolve()),
        "selection_sha256": SELECTION_SHA256,
        "training_sha256": selection["training_sha256"],
        "selected_heads": selection["selections"],
        "selection_source_sha256": selection["source"],
        "model_path": str(Path(model_path).resolve()),
        "model_sha256": model_hashes,
        "protocol": protocol(),
        "protocol_sha256": json_sha(protocol()),
        "source_sha256": {
            "reference_train": sha(ROOT / "examples/local_async/train.py"),
            "head_quality_driver": sha(ROOT / "scripts/verify_head_quality.py"),
            "fresh_holdout_driver": sha(__file__),
        },
        "created_unix_seconds": time.time(),
    }
    receipt_path = output / "freeze.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    digest = sha(receipt_path)
    (output / "freeze.sha256").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"freeze": str(receipt_path), "freeze_sha256": digest, "pairs": PAIR_COUNT}))


def _validate_frozen(output, expected_sha):
    output = Path(output).resolve()
    frozen_path = output / "freeze.json"
    if sha(frozen_path) != expected_sha or (output / "freeze.sha256").read_text().strip() != expected_sha:
        raise ValueError("Frozen protocol hash does not match reviewed freeze SHA256")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["selection_sha256"] != SELECTION_SHA256:
        raise ValueError("Frozen selection identity changed")
    if frozen["protocol"] != protocol() or frozen["protocol_sha256"] != json_sha(protocol()):
        raise ValueError("Fresh pair protocol was changed")
    source_paths = {
        "reference_train": ROOT / "examples/local_async/train.py",
        "head_quality_driver": ROOT / "scripts/verify_head_quality.py",
        "fresh_holdout_driver": Path(__file__),
    }
    for name, path in source_paths.items():
        if sha(path) != frozen["source_sha256"][name]:
            raise ValueError(f"Frozen evaluation source changed: {name}")
    selection, training, model_hashes = _check_old_receipts(frozen["selection_dir"], frozen["model_path"])
    if selection["training_sha256"] != frozen["training_sha256"] or selection["selections"] != frozen["selected_heads"]:
        raise ValueError("Frozen selected-head inventory changed")
    if model_hashes != frozen["model_sha256"]:
        raise ValueError("Frozen model assets changed")
    return output, frozen, selection, training


def _check_evaluation_time(started, stage):
    if time.perf_counter() - started > MAX_EVALUATION_SECONDS:
        raise TimeoutError(f"Fresh holdout exceeded {MAX_EVALUATION_SECONDS}s during {stage}")


def _verify_oracle_prompts(reference, model, tokenizer, prompts, by_text, features, token_ids, started):
    projections = {str(count): {"max_ulp": 0.0, "max_absolute_error": 0.0} for count in (2, 10)}
    for prompt in prompts:
        _check_evaluation_time(started, "cached-logit oracle prompt")
        check = reference._verify_cached_bf16_logits(model, tokenizer, [prompt], by_text, features, token_ids)
        for count, values in check["projections"].items():
            projections[count]["max_ulp"] = max(projections[count]["max_ulp"], values["max_ulp"])
            projections[count]["max_absolute_error"] = max(
                projections[count]["max_absolute_error"], values["max_absolute_error"]
            )
        _check_evaluation_time(started, "cached-logit oracle prompt")
    return {
        "max_ulp": max(item["max_ulp"] for item in projections.values()),
        "max_absolute_error": max(item["max_absolute_error"] for item in projections.values()),
        "projections": projections,
        "tolerance_ulp": 1.0,
    }


def _evaluate(output, expected_sha):
    started = time.perf_counter()
    output, frozen, selection, training = _validate_frozen(output, expected_sha)
    if (output / "test.json").exists() or (output / "failure.json").exists():
        raise FileExistsError("Refusing to overwrite an evaluation or failure receipt")
    if shutil.disk_usage(output).free < MIN_FREE_DISK_BYTES:
        raise OSError("Fresh holdout requires at least 10 GiB of free disk")

    import torch
    from safetensors.torch import load_file

    from examples.local_async import train as reference
    from scripts import verify_head_quality as quality

    torch.set_num_threads(CPU_THREADS)
    selection_dir = Path(frozen["selection_dir"])
    if selection["training_sha256"] != quality.sha(selection_dir / "training.json"):
        raise ValueError("Training receipt changed after freeze")
    quality._validate_inventory(training["runs"])
    if {run: quality._select(data) for run, data in training["runs"].items()} != selection["selections"]:
        raise ValueError("Validation-only frozen selection does not recompute exactly")
    # No new prompt text or model forward is touched before the selection checks above.
    if reference._model_hashes(Path(frozen["model_path"])) != training["manifest"]["prior_model_sha256"]:
        raise ValueError("Model assets differ from frozen preparation")

    study_dir = Path(selection["input_paths"]["features"]).resolve().parent
    _, _, old_prompts, old_features, _ = quality._features(study_dir)
    model_path = Path(frozen["model_path"])
    torch.cuda.reset_peak_memory_stats()
    model = (
        reference.AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    tokenizer = reference.AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model.requires_grad_(False)
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone(), requires_grad=False)
    model.config.tie_word_embeddings = False
    initial_artifact = load_file(str(selection_dir / "initial-head.safetensors"), device="cpu")
    token_ids = torch.tensor([tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)], device="cuda")
    if token_ids.cpu().tolist() != initial_artifact["token_ids"].tolist():
        raise ValueError("Tokenizer digit IDs differ from frozen initial head")
    initial_rows = initial_artifact["initial_head"].to(torch.bfloat16)
    if not torch.equal(model.lm_head.weight[token_ids].detach().cpu(), initial_rows):
        raise ValueError("Cached model digit rows differ from frozen initial head")

    pairs = frozen["protocol"]["pairs"]
    prompts_by_pair = [reference._prompts(a, b) for a, b in pairs]
    unique_prompts = list(dict.fromkeys(prompt for first, answers in prompts_by_pair for prompt in (first, *answers)))
    prompts, features = dict(old_prompts), dict(old_features)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for prompt in unique_prompts:
            _check_evaluation_time(inference_started, "backbone feature prompt")
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
            hidden = model.model(**encoded, use_cache=False).last_hidden_state[0, -1].float().cpu().contiguous()
            key = f"fresh-{len(features):04d}"
            prompts[key], features[key] = prompt, hidden
            _check_evaluation_time(inference_started, "backbone feature prompt")
    by_text = reference._prompt_keys(prompts)
    cuda_features = {key: value.to("cuda") for key, value in features.items()}
    oracle_pairs = [pairs[index] for index in frozen["protocol"]["oracle_pair_indices"]]
    oracle_prompts = list(
        dict.fromkeys(
            prompt for a, b in oracle_pairs for prompt in (reference._prompts(a, b)[0], *reference._prompts(a, b)[1])
        )
    )
    heads = {"initial": initial_rows}
    for run, candidate in selection["selections"].items():
        heads[run] = (
            load_file(str(selection_dir / candidate["checkpoint"]), device="cpu")["installed_bf16_rows"]
            if candidate["checkpoint"]
            else initial_rows
        )
    for name, rows in heads.items():
        if rows.shape != initial_rows.shape or not torch.isfinite(rows).all():
            raise ValueError(f"Invalid or nonfinite digit head rows: {name}")

    results, oracle_checks = {}, {}
    output_path = output / "test.json"
    try:
        for name, rows in heads.items():
            _check_evaluation_time(inference_started, f"head {name}")
            with torch.no_grad():
                model.lm_head.weight.index_copy_(0, token_ids, rows.to("cuda"))
            oracle_checks[name] = _verify_oracle_prompts(
                reference, model, tokenizer, oracle_prompts, by_text, cuda_features, token_ids, inference_started
            )
            evaluated = reference._evaluate_study(model, rows.to("cuda"), pairs, by_text, cuda_features)
            _check_evaluation_time(inference_started, f"head {name}")
            per_pair = evaluated["per_pair"]
            if len(per_pair) != PAIR_COUNT or [entry["pair"] for entry in per_pair] != pairs:
                raise ValueError(f"Incomplete per-pair evaluation for {name}")
            if not all(
                math.isfinite(entry[key]) for entry in per_pair for key in ("expected_reward", "answer_success")
            ):
                raise ValueError(f"Nonfinite per-pair metric for {name}")
            results[name] = evaluated
        receipt = {
            "freeze_sha256": expected_sha,
            "selection_sha256": frozen["selection_sha256"],
            "partition": "256 fresh unordered operand pairs in [10, 99]",
            "claim_scope": frozen["claim_scope"],
            "execution": "frozen CPU-trained head evaluation on cached local BF16 backbone",
            "oracle_pair_indices": frozen["protocol"]["oracle_pair_indices"],
            "oracle_pair_count": len(oracle_pairs),
            "oracle_checks": oracle_checks,
            "results": results,
            "runtime": {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "torch_threads": torch.get_num_threads(),
                "cuda_runtime": torch.version.cuda,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                "peak_rss_bytes": quality.peak_rss_bytes(),
            },
        }
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
    except Exception as exc:
        error = {
            "freeze_sha256": expected_sha,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes": quality.peak_rss_bytes(),
        }
        if torch.cuda.is_initialized():
            error["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            error["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        with (output / "failure.json").open("x", encoding="utf-8") as stream:
            json.dump(error, stream, indent=2)
        raise


def evaluate(output, expected_sha):
    started = time.perf_counter()
    output = Path(output).resolve()
    _validate_frozen(output, expected_sha)
    if (output / "test.json").exists() or (output / "failure.json").exists():
        raise FileExistsError("Refusing to overwrite an evaluation or failure receipt")
    try:
        _evaluate(output, expected_sha)
    except Exception as exc:
        if not (output / "failure.json").exists() and not (output / "test.json").exists():
            error = {
                "freeze_sha256": expected_sha,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            }
            torch = sys.modules.get("torch")
            quality = sys.modules.get("scripts.verify_head_quality")
            if quality is not None:
                error["peak_rss_bytes"] = quality.peak_rss_bytes()
            if torch is not None and torch.cuda.is_initialized():
                error["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                error["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
            with (output / "failure.json").open("x", encoding="utf-8") as stream:
                json.dump(error, stream, indent=2)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("freeze")
    prepare_parser.add_argument("--selection-dir", required=True)
    prepare_parser.add_argument("--model", required=True)
    prepare_parser.add_argument("--output", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.selection_dir, args.model, args.output)
    else:
        evaluate(args.output, args.freeze_sha256)


if __name__ == "__main__":
    main()
