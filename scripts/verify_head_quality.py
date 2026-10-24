"""Bounded CPU head-quality comparator and hash-checked held-out evaluation."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.local_async import train as reference  # noqa: E402

SEEDS = (11, 23, 37, 51, 71)
CHECKPOINTS = (32, 128, 256)
CONFIG = {
    "seeds": list(SEEDS),
    "updates": 256,
    "groups_per_update": 2,
    "checkpoints": list(CHECKPOINTS),
    "clip_arm": 0.5,
    "baseline_arm": None,
    "lr": 0.005,
    "training": "CPU one-thread BF16 forward, FP32 master",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def peak_rss_bytes():
    if sys.platform == "win32":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize
    if sys.platform == "darwin":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _features(study):
    study = Path(study)
    manifest_path, features_path, prompts_path = (
        study / name for name in ("manifest.json", "features.safetensors", "prompts.json")
    )
    prep = json.loads((study / "preparation.json").read_text())
    for key, path in (
        ("manifest_sha256", manifest_path),
        ("features_sha256", features_path),
        ("prompts_sha256", prompts_path),
    ):
        if prep[key] != sha(path):
            raise ValueError(f"Cached study hash mismatch: {path.name}")
    manifest, prompts = json.loads(manifest_path.read_text()), json.loads(prompts_path.read_text())
    features = load_file(str(features_path), device="cpu")
    if set(prompts) != set(features) or set(prompts.values()) != set(
        reference._study_partition_prompts(manifest, "train", "validation")
    ):
        raise ValueError("Cached train/validation feature inventory mismatch")
    return study, manifest, prompts, features, prep


def _load_initial(path):
    path = Path(path)
    if path.is_dir():
        raise ValueError("--initial-checkpoint must be an explicit checkpoint file")
    tensors = load_file(str(path), device="cpu")
    if not {"head_weights", "initial_head", "token_ids"}.issubset(tensors):
        raise ValueError("Initial checkpoint lacks the native head tensors")
    receipt_path = path.parent / "run_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("Initial checkpoint must remain beside its native run receipt")
    run_receipt = json.loads(receipt_path.read_text())
    match = next((item for item in run_receipt["checkpoint_receipts"].values() if item["filename"] == path.name), None)
    if not match or match["sha256"] != sha(path):
        raise ValueError("Initial checkpoint does not match its native run receipt")
    if not torch.isfinite(tensors["initial_head"]).all():
        raise ValueError("Initial checkpoint contains a nonfinite head")
    if tensor_sha(tensors["initial_head"]) != match["initial_head_sha256"]:
        raise ValueError("Initial head does not match its native checkpoint receipt")
    return tensors["initial_head"].float(), tensors["token_ids"].long(), receipt_path


def _sample(prompt, count, version, rows, features, by_text, generator):
    hidden = features[by_text[prompt]].float()
    with torch.no_grad():
        logits = torch.nn.functional.linear(hidden.to(torch.bfloat16), rows[:count].to(torch.bfloat16)).float()
        dist = torch.distributions.Categorical(logits=logits)
        index = int(torch.multinomial(dist.probs, 1, generator=generator))
    return reference.Action(hidden, index, float(dist.log_prob(torch.tensor(index))), version)


def _group(index, version, seed, schedule, rows, features, by_text):
    uid = f"{index:04d}"
    generator = torch.Generator(device="cpu").manual_seed(reference.group_seed(seed, uid))
    a, b = schedule[index % len(schedule)]
    first, answers = reference._prompts(a, b)
    tool_actions = [_sample(first, 2, version, rows, features, by_text, generator) for _ in range(2)]
    actions, rewards = [], []
    for tool_action in tool_actions:
        answer = _sample(answers[tool_action.index], 10, version, rows, features, by_text, generator)
        actions.extend((tool_action, answer))
        rewards.append(float(answer.index == (a + b) % 10) + 0.25 * (tool_action.index == 0))
    return reference.Group(uid, version, actions, rewards)


def _cpu_validate(rows, pairs, by_text, features):
    # Preserve the production evaluator's arithmetic and half-precision projection.
    model = SimpleNamespace(lm_head=SimpleNamespace(weight=rows.to(torch.bfloat16)))
    return reference._evaluate_study(model, model.lm_head.weight, pairs, by_text, features)


def train(study_path, initial_path, output):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    study, manifest, prompts, features, prep = _features(study_path)
    initial, ids, initial_receipt_path = _load_initial(Path(initial_path))
    if initial.shape[0] != 10 or ids.numel() != 10:
        raise ValueError("Expected the ten decimal output rows")
    by_text = reference._prompt_keys(prompts)
    source_hashes = {"train.py": sha(Path(reference.__file__)), "driver.py": sha(__file__)}
    input_hashes = {
        "manifest": sha(study / "manifest.json"),
        "features": sha(study / "features.safetensors"),
        "prompts": sha(study / "prompts.json"),
        "preparation": sha(study / "preparation.json"),
        "initial_checkpoint": sha(initial_path),
        "initial_run_receipt": sha(initial_receipt_path),
    }
    initial_file = output / "initial-head.safetensors"
    save_file({"initial_head": initial, "token_ids": ids}, str(initial_file))
    input_hashes["saved_initial_head"] = sha(initial_file)
    input_paths = {
        "manifest": str(study / "manifest.json"),
        "features": str(study / "features.safetensors"),
        "prompts": str(study / "prompts.json"),
        "preparation": str(study / "preparation.json"),
        "initial_checkpoint": str(Path(initial_path).resolve()),
        "initial_run_receipt": str(initial_receipt_path.resolve()),
        "saved_initial_head": str(initial_file.resolve()),
    }
    metadata = {
        "schema": 1,
        "config": CONFIG,
        "inputs": input_hashes,
        "source": source_hashes,
        "prior_model_sha256": prep.get("model_sha256"),
        "partition_scope": "train and validation only",
        "test_accessed": False,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": sys.modules["transformers"].__version__,
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    all_metrics = {}
    ids_model = torch.arange(10)
    installed = torch.zeros_like(initial, dtype=torch.bfloat16)
    model = SimpleNamespace(lm_head=SimpleNamespace(weight=installed))
    for seed in SEEDS:
        schedule = reference.study_schedule(seed)
        for arm, clip in (("clipped", 0.5), ("unclipped", None)):
            head = initial.clone().requires_grad_(True)
            optimizer = torch.optim.SGD([head], lr=CONFIG["lr"])
            with torch.no_grad():
                installed.copy_(head.to(torch.bfloat16))
            before = _cpu_validate(head, manifest["partitions"]["validation"], by_text, features)
            rows = []
            arm_started = time.perf_counter()
            for step in range(1, 257):
                batch = [
                    _group((step - 1) * 2 + j, step - 1, seed, schedule, installed, features, by_text) for j in range(2)
                ]
                loss, grad_norm = reference.apply_policy_update(
                    batch, head, optimizer, model, ids_model, training_dtype=torch.bfloat16, max_grad_norm=clip
                )
                rows.append({"step": step, "loss": loss, "preclip_grad_norm": grad_norm})
                if step in CHECKPOINTS:
                    validation = _cpu_validate(head.detach(), manifest["partitions"]["validation"], by_text, features)
                    cp = output / f"{arm}-seed-{seed}-step-{step}.safetensors"
                    save_file(
                        {
                            "head_weights": head.detach().cpu(),
                            "installed_bf16_rows": installed.cpu(),
                            "initial_head": initial.cpu(),
                            "token_ids": ids.cpu(),
                        },
                        str(cp),
                    )
                    restored = load_file(str(cp), device="cpu")
                    if not torch.equal(restored["installed_bf16_rows"], restored["head_weights"].to(torch.bfloat16)):
                        raise ValueError("Checkpoint master/installed rows do not round trip")
                    rows.append(
                        {
                            "step": step,
                            "validation": validation,
                            "checkpoint": cp.name,
                            "sha256": sha(cp),
                            "head_sha256": tensor_sha(restored["head_weights"]),
                        }
                    )
            all_metrics[f"{arm}-seed-{seed}"] = {
                "initial_validation": before,
                "updates": rows,
                "elapsed_seconds": time.perf_counter() - arm_started,
            }
    _validate_inventory(all_metrics)
    metadata["input_paths"] = input_paths
    metadata["peak_rss_bytes"] = peak_rss_bytes()
    metadata["wall_seconds_before_receipt_write"] = time.perf_counter() - started
    receipt = {"manifest": metadata, "runs": all_metrics}
    (output / "training.json").write_text(json.dumps(receipt, indent=2))
    # Freeze selection using validation expected reward; earliest step wins ties.
    selection = {run: _select(result) for run, result in all_metrics.items()}
    selected = {
        "schema": 1,
        "training_sha256": sha(output / "training.json"),
        "inputs": input_hashes,
        "input_paths": input_paths,
        "config": CONFIG,
        "source": source_hashes,
        "selections": selection,
        "test_accessed": False,
    }
    with (output / "selection.json").open("x") as stream:
        json.dump(selected, stream, indent=2)


def test(study_path, model_path, output):
    started = time.perf_counter()
    output = Path(output)
    output_path = output / "test.json"
    if output_path.exists():
        raise FileExistsError("Refusing to overwrite test receipt")
    selection_path = output / "selection.json"
    receipt = json.loads(selection_path.read_text())
    if receipt["test_accessed"] or receipt["training_sha256"] != sha(output / "training.json"):
        raise ValueError("Frozen CPU selection/training receipt hash mismatch")
    if receipt["config"] != CONFIG:
        raise ValueError("Comparator configuration differs from frozen training")
    if Path(study_path).resolve() != Path(receipt["input_paths"]["features"]).parent.resolve():
        raise ValueError("--features-from must match the frozen training cache path")
    for name, digest in receipt["inputs"].items():
        path = Path(receipt["input_paths"][name])
        if sha(path) != digest:
            raise ValueError(f"Test input hash mismatch: {path.name}")
    for name, digest in receipt["source"].items():
        path = Path(reference.__file__) if name == "train.py" else Path(__file__)
        if sha(path) != digest:
            raise ValueError(f"Source hash mismatch: {name}")
    for candidate in receipt["selections"].values():
        if candidate["checkpoint"] and sha(output / candidate["checkpoint"]) != candidate["sha256"]:
            raise ValueError("Selected checkpoint hash mismatch")
    # Recompute the validation-only winner from the hashed training receipt.
    training = json.loads((output / "training.json").read_text())
    _validate_inventory(training["runs"])
    if training["manifest"]["config"] != CONFIG:
        raise ValueError("Training configuration hash mismatch")
    if reference._model_hashes(Path(model_path)) != training["manifest"]["prior_model_sha256"]:
        raise ValueError("Cached pretrained model assets differ from training preparation")
    for run_data in training["runs"].values():
        for item in run_data["updates"]:
            if "checkpoint" in item and sha(output / item["checkpoint"]) != item["sha256"]:
                raise ValueError("Candidate checkpoint hash mismatch")
    if {run: _select(run_data) for run, run_data in training["runs"].items()} != receipt["selections"]:
        raise ValueError("Frozen selection does not match validation results")
    # All frozen-selection checks above precede the first test partition/prompt read.
    study, manifest, prompts, features, prep = _features(study_path)
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
    initial_tensors = load_file(str(output / "initial-head.safetensors"), device="cpu")
    token_ids = torch.tensor([tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)], device="cuda")
    if token_ids.cpu().tolist() != initial_tensors["token_ids"].tolist():
        raise ValueError("Tokenizer digit IDs differ from the frozen initial checkpoint")
    initial_rows = initial_tensors["initial_head"].to(torch.bfloat16)
    if not torch.equal(model.lm_head.weight[token_ids].detach().cpu(), initial_rows):
        raise ValueError("Cached model digit rows differ from the frozen initial head")
    test_prompts = reference._study_partition_prompts(manifest, "test")
    with torch.inference_mode():
        for prompt in test_prompts:
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
            hidden = model.model(**encoded, use_cache=False).last_hidden_state[0, -1].float().cpu()
            key = f"heldout-{len(features):03d}"
            prompts[key], features[key] = prompt, hidden
    by_text = reference._prompt_keys(prompts)
    results, logit_checks = {}, {}
    with torch.no_grad():
        model.lm_head.weight.index_copy_(0, token_ids, initial_rows.to("cuda"))
    cuda_features = {key: value.to("cuda") for key, value in features.items()}
    full_model_logits = reference._verify_cached_bf16_logits(
        model, tokenizer, test_prompts, by_text, cuda_features, token_ids
    )
    initial_test = reference._evaluate_study(
        model, initial_rows.to("cuda"), manifest["partitions"]["test"], by_text, cuda_features
    )
    for run, candidate in receipt["selections"].items():
        rows = (
            load_file(str(output / candidate["checkpoint"]), device="cpu")["installed_bf16_rows"]
            if candidate["checkpoint"]
            else initial_rows
        )
        with torch.no_grad():
            model.lm_head.weight.index_copy_(0, token_ids, rows.to("cuda"))
        logit_checks[run] = reference._verify_cached_bf16_logits(
            model, tokenizer, test_prompts, by_text, cuda_features, token_ids
        )
        results[run] = reference._evaluate_study(
            model, rows.to("cuda"), manifest["partitions"]["test"], by_text, cuda_features
        )
    with output_path.open("x") as stream:
        json.dump(
            {
                "selection_sha256": sha(selection_path),
                "partition": "reused heldout set",
                "execution": "CPU training, GPU evaluation",
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cached_logit_oracle": full_model_logits,
                "initial_test": initial_test,
                "selection_logit_checks": logit_checks,
                "results": results,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                "peak_rss_bytes": peak_rss_bytes(),
            },
            stream,
            indent=2,
        )


def _validate_inventory(runs):
    expected = {f"{arm}-seed-{seed}" for arm in ("clipped", "unclipped") for seed in SEEDS}
    if set(runs) != expected:
        raise ValueError("Expected all five paired clipped/unclipped runs")
    for run in runs.values():
        steps = [item["step"] for item in run["updates"] if "loss" in item]
        checkpoints = [item["step"] for item in run["updates"] if "validation" in item]
        if steps != list(range(1, CONFIG["updates"] + 1)) or checkpoints != list(CHECKPOINTS):
            raise ValueError("Incomplete or duplicate update/checkpoint inventory")


def _select(run_data):
    candidates = [{"step": 0, **run_data["initial_validation"], "checkpoint": None}]
    candidates += [
        {**item["validation"], "step": item["step"], "checkpoint": item["checkpoint"], "sha256": item["sha256"]}
        for item in run_data["updates"]
        if "validation" in item
    ]
    return max(candidates, key=lambda item: (item["expected_reward"], -item["step"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--features-from", required=True)
    train_parser.add_argument("--initial-checkpoint", required=True)
    train_parser.add_argument("--output", required=True)
    test_parser = sub.add_parser("test")
    test_parser.add_argument("--features-from", required=True)
    test_parser.add_argument("--model", required=True)
    test_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.features_from, args.initial_checkpoint, args.output)
    else:
        test(args.features_from, args.model, args.output)


if __name__ == "__main__":
    main()
