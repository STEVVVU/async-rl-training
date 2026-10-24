"""Bounded pretrained policy-gradient reference in an owned arithmetic tool environment."""

import asyncio
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from skyrl.train.staleness import GroupDecision, StalenessFence

STUDY_SEEDS = (11, 23, 37, 51, 71)
STUDY_CHECKPOINTS = (32, 128, 256)
STUDY_GROUPS_PER_UPDATE = 2
STUDY_UPDATES = 256
STUDY_ACTIONS_PER_GROUP = 4
STUDY_BF16_ULP_FLOOR = 1e-42
MAX_GRAD_NORM = 0.5
STUDY_SOURCE_FILES = ("examples/local_async/train.py", "scripts/verify_async_rl.py")


def study_manifest():
    pairs = [(a, b) for a in range(10) for b in range(a, 10)]
    random.Random(421).shuffle(pairs)
    return {
        "schema": 1,
        "shuffle_seed": 421,
        "partitions": {
            "train": [list(pair) for pair in pairs[:35]],
            "validation": [list(pair) for pair in pairs[35:45]],
            "test": [list(pair) for pair in pairs[45:]],
        },
    }


def study_schedule(seed):
    pairs = study_manifest()["partitions"]["train"]
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    return [pairs[order[i % len(order)]] for i in range(STUDY_UPDATES * STUDY_GROUPS_PER_UPDATE)]


def study_run_order():
    return [
        {"arm": arm, "seed": seed, "index": i * 2 + j}
        for i, seed in enumerate(STUDY_SEEDS)
        for j, arm in enumerate(("sync", "async") if i % 2 == 0 else ("async", "sync"))
    ]


def study_config():
    return {
        "updates": STUDY_UPDATES,
        "groups_per_update": STUDY_GROUPS_PER_UPDATE,
        "action_tokens_per_group": STUDY_ACTIONS_PER_GROUP,
        "learning_rate": 0.005,
        "baseline": 0.5,
        "ratio_clip": [0.8, 1.2],
        "max_grad_norm": MAX_GRAD_NORM,
        "max_age": 1,
        "sampling_precision": "bfloat16 installed model rows and bfloat16 features; float32 logits",
        "master_precision": "float32 head; differentiable bfloat16 training forward",
        "rng": "sha256(seed,group_uid) CUDA generator",
    }


def record_action_tokens(totals, per_uid, uid, count):
    totals["action_tokens"] += count
    per_uid[uid] = per_uid.get(uid, 0) + count


def _study_partition_prompts(manifest, *partitions):
    pairs = [pair for partition in partitions for pair in manifest["partitions"][partition]]
    return _study_prompts(pairs)


def validate_study_metrics(metrics, arm, seed, order_index):
    counts = metrics["counts"]
    versions = metrics["group_versions"]
    records = metrics["terminal_records"]
    prompt_tokens = metrics["prompt_tokens_by_uid"]
    action_tokens = metrics["action_tokens_by_uid"]
    accepted_count = STUDY_UPDATES * STUDY_GROUPS_PER_UPDATE
    accepted = {f"{index:04d}" for index in range(accepted_count)}
    tails = (
        {f"{index:04d}" for index in range(accepted_count, accepted_count + STUDY_GROUPS_PER_UPDATE)}
        if arm == "async"
        else set()
    )
    expected_uids = accepted | tails
    expected_generated = 514 if arm == "async" else 512
    expected_actions = expected_generated * STUDY_ACTIONS_PER_GROUP
    expected_counts = {
        "generated": expected_generated,
        "accepted": accepted_count,
        "stale": 0,
        "tail": len(tails),
        "action_tokens": expected_actions,
        "mixed": accepted_count - STUDY_GROUPS_PER_UPDATE if arm == "async" else 0,
    }
    if metrics["run_id"] != f"{arm}-seed-{seed}" or metrics["run_order_index"] != order_index:
        raise ValueError("Run identity/order receipt mismatch")
    if metrics["updates"] != STUDY_UPDATES or metrics["arm"] != arm or metrics["seed"] != seed:
        raise ValueError("Run update count or paired identity mismatch")
    if set(versions) != expected_uids or set(records) != expected_uids:
        raise ValueError("Generated group UID ledger is incomplete or contains unexpected UIDs")
    for name, expected in expected_counts.items():
        if counts.get(name) != expected:
            raise ValueError(f"Run count invariant failed for {name}: {counts.get(name)} != {expected}")
    if {uid for uid, record in records.items() if record["decision"] == "keep"} != accepted:
        raise ValueError("Run does not contain 512 unique accepted groups")
    if set(prompt_tokens) != expected_uids or set(action_tokens) != expected_uids:
        raise ValueError("Per-group token ledger is incomplete")
    if any(count != STUDY_ACTIONS_PER_GROUP for count in action_tokens.values()):
        raise ValueError("Per-group action-token budget changed")
    if sum(action_tokens.values()) != counts["action_tokens"] or sum(prompt_tokens.values()) != counts["prompt_tokens"]:
        raise ValueError("Per-group token ledger does not match aggregate token counts")
    for uid in accepted:
        group_index = int(uid)
        update_version = group_index // STUDY_GROUPS_PER_UPDATE
        actual = versions[uid]
        expected = (
            [update_version] * 4
            if arm == "sync" or update_version == 0
            else [update_version - 1] * 2 + [update_version] * 2
        )
        if actual != expected:
            raise ValueError(f"Unexpected action policy versions for {uid}: {actual}")
        record = records[uid]
        age = record["consumer_version"] - min(actual)
        if (
            record["decision"] != "keep"
            or record["versions"] != actual
            or record["oldest"] != min(actual)
            or record["consumer_version"] != update_version
            or age < 0
            or age > 1
        ):
            raise ValueError(f"Fence ledger invariant failed for {uid}")
        if arm == "sync" and (age != 0 or len(set(actual)) != 1):
            raise ValueError(f"Synchronous group is not current-version for {uid}")
    for uid in tails:
        if (
            records[uid]["decision"] != "tail"
            or records[uid]["consumer_version"] != STUDY_UPDATES
            or records[uid]["oldest"] != STUDY_UPDATES - 1
            or versions[uid] != [STUDY_UPDATES - 1] * 2 + [STUDY_UPDATES] * 2
        ):
            raise ValueError(f"Async tail ledger invariant failed for {uid}")
    finite_values = [
        *metrics["losses"],
        metrics["initial_validation"]["expected_reward"],
        metrics["initial_validation"]["answer_success"],
    ]
    finite_values.extend(
        value
        for pair in metrics["initial_validation"]["per_pair"]
        for value in (pair["expected_reward"], pair["answer_success"])
    )
    finite_values.extend(
        item[metric] for item in metrics["validation"] for metric in ("expected_reward", "answer_success")
    )
    finite_values.extend(
        value
        for item in metrics["validation"]
        for pair in item["per_pair"]
        for value in (pair["expected_reward"], pair["answer_success"])
    )
    finite_values.extend(metrics["timings"].values())
    if len(metrics["losses"]) != STUDY_UPDATES or len(metrics["validation"]) != len(STUDY_CHECKPOINTS):
        raise ValueError("Loss or validation checkpoint ledger is incomplete")
    if [item.get("step") for item in metrics["validation"]] != list(STUDY_CHECKPOINTS):
        raise ValueError("Validation checkpoint sequence is incomplete or reordered")
    if set(metrics.get("consistency_checks", {})) != {"initial", *(str(step) for step in STUDY_CHECKPOINTS)}:
        raise ValueError("BF16 cached-logit checks are incomplete")
    if any(item.get("max_ulp", math.inf) > 1.0 for item in metrics["consistency_checks"].values()):
        raise ValueError("BF16 cached-logit check exceeded the predeclared one-ULP tolerance")
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("Nonfinite study result")
    return True


def validate_selection_run(metrics, receipt, entry, manifest_sha256, source_sha256, model_sha256, data_sha256):
    run_id = f"{entry['arm']}-seed-{entry['seed']}"
    steps = [str(step) for step in STUDY_CHECKPOINTS]
    if receipt.get("run_id") != run_id or receipt.get("run_order_index") != entry["index"]:
        raise ValueError(f"Run receipt identity mismatch: {run_id}")
    if metrics.get("run_id") != run_id or metrics.get("run_order_index") != entry["index"]:
        raise ValueError(f"Run metrics identity mismatch: {run_id}")
    if not receipt.get("run_passed") or not metrics.get("run_passed"):
        raise ValueError(f"Run did not pass study invariants: {run_id}")
    if receipt.get("manifest_sha256") != manifest_sha256 or metrics.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"Run manifest identity mismatch: {run_id}")
    if receipt.get("source_sha256") != source_sha256 or metrics.get("source_sha256") != source_sha256:
        raise ValueError(f"Run source identity mismatch: {run_id}")
    if receipt.get("model_sha256") != model_sha256 or metrics.get("model_sha256") != model_sha256:
        raise ValueError(f"Run model identity mismatch: {run_id}")
    if metrics.get("data_sha256") != data_sha256:
        raise ValueError(f"Run prepared-data identity mismatch: {run_id}")
    if receipt.get("config") != study_config() or metrics.get("config") != study_config():
        raise ValueError(f"Run configuration mismatch: {run_id}")
    if any(
        receipt.get(key) != metrics.get(key)
        for key in (
            "arm",
            "seed",
            "updates",
            "counts",
            "schedule_sha256",
            "data_sha256",
            "runtime",
            "model_config",
            "trainable_parameters",
            "training_policy",
            "runtime_policy",
        )
    ):
        raise ValueError(f"Run receipt counts, schedule, or data identity mismatch: {run_id}")
    expected_schedule_sha256 = hashlib.sha256(
        json.dumps(study_schedule(entry["seed"]), separators=(",", ":")).encode()
    ).hexdigest()
    if metrics.get("schedule_sha256") != expected_schedule_sha256:
        raise ValueError(f"Run schedule identity mismatch: {run_id}")
    if [item.get("step") for item in metrics.get("validation", [])] != list(STUDY_CHECKPOINTS):
        raise ValueError(f"Run does not have all three validation checkpoints: {run_id}")
    if set(metrics.get("checkpoint_receipts", {})) != set(steps) or set(receipt.get("checkpoint_receipts", {})) != set(
        steps
    ):
        raise ValueError(f"Run does not have all three checkpoint receipts: {run_id}")
    if metrics.get("arm") != entry["arm"] or metrics.get("seed") != entry["seed"]:
        raise ValueError(f"Run arm/seed identity mismatch: {run_id}")
    for step in steps:
        if metrics["checkpoint_receipts"][step] != receipt["checkpoint_receipts"][step]:
            raise ValueError(f"Run checkpoint receipt mismatch at {step}: {run_id}")
        cp = metrics["checkpoint_receipts"][step]
        if cp.get("filename") != f"step-{step}.safetensors" or not all(
            cp.get(key) for key in ("sha256", "head_sha256", "installed_bf16_sha256", "initial_head_sha256")
        ):
            raise ValueError(f"Incomplete checkpoint identity at {step}: {run_id}")
    validate_study_metrics(metrics, entry["arm"], entry["seed"], entry["index"])
    return True


def _validate_run_inventory(run_root, order):
    expected = {f"{entry['arm']}-seed-{entry['seed']}" for entry in order}
    actual = {path.name for path in run_root.iterdir() if path.is_dir()}
    if actual != expected or len(order) != len(STUDY_SEEDS) * 2:
        raise ValueError("Study must contain exactly all ten paired runs")
    return True


def group_seed(seed, uid):
    digest = hashlib.sha256(f"{seed}:{uid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def bf16_logit_error(actual, cached):
    actual, cached = actual.to(torch.bfloat16), cached.to(torch.bfloat16)
    if not torch.isfinite(actual).all() or not torch.isfinite(cached).all():
        raise ValueError("Nonfinite BF16 logit in cached-feature consistency check")
    actual32, cached32 = actual.float(), cached.float()
    upper = torch.nextafter(actual, torch.full_like(actual, float("inf"))).float()
    lower = torch.nextafter(actual, torch.full_like(actual, float("-inf"))).float()
    upper_gap = torch.where(torch.isfinite(upper), (upper - actual32).abs(), torch.zeros_like(actual32))
    lower_gap = torch.where(torch.isfinite(lower), (actual32 - lower).abs(), torch.zeros_like(actual32))
    ulp = torch.maximum(upper_gap, lower_gap)
    ulp = torch.where(actual32 == 0, ulp.clamp_min(STUDY_BF16_ULP_FLOOR), ulp)
    error = (actual32 - cached32).abs()
    return float((error / ulp).max()), float(error.max())


def _model_hashes(model_path):
    return {
        path.relative_to(model_path).as_posix(): _sha256(path)
        for path in sorted(model_path.rglob("*"))
        if path.is_file()
    }


def _source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {name: _sha256(root / name) for name in STUDY_SOURCE_FILES}


def _runtime_identity():
    import transformers

    return {
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }


def apply_policy_update(batch, head, optimizer, model, ids, training_dtype=None, max_grad_norm=MAX_GRAD_NORM):
    terms = []
    for group in batch:
        for i, action in enumerate(group.actions):
            count = 2 if i % 2 == 0 else 10
            hidden = action.hidden if training_dtype is None else action.hidden.to(training_dtype)
            weights = head[:count] if training_dtype is None else head[:count].to(training_dtype)
            logits = torch.nn.functional.linear(hidden, weights)
            if training_dtype is not None:
                logits = logits.float()
            logprob = torch.log_softmax(logits, -1)[action.index]
            ratio = torch.exp(logprob - action.logprob)
            advantage = group.rewards[i // 2] - 0.5
            terms.append(-torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage))
    loss = torch.stack(terms).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite policy loss")
    optimizer.zero_grad()
    loss.backward()
    if not torch.isfinite(head.grad).all():
        raise RuntimeError("Nonfinite policy gradient")
    grad_norm = float(head.grad.norm())
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(head, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    if not torch.isfinite(head).all():
        raise RuntimeError("Nonfinite FP32 master head")
    with torch.no_grad():
        model.lm_head.weight.index_copy_(0, ids, head.to(model.lm_head.weight.dtype))
    return float(loss.detach()), grad_norm


def _prompts(a, b):
    first = f"Compute ({a}+{b}) modulo 10. Choose tool 0 for sum, 1 for difference. Tool:"
    return first, [first + f" {choice}\nTool result: {tool(a, b, choice)}\nAnswer digit:" for choice in (0, 1)]


def _study_prompts(pairs):
    prompts = []
    for a, b in pairs:
        first, answers = _prompts(a, b)
        prompts.extend((first, *answers))
    return list(dict.fromkeys(prompts))


def prepare_study(model_path: Path, output: Path):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(15)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            use_safetensors=True,
        )
        .to("cuda")
        .eval()
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    manifest = study_manifest()
    manifest["model_revision"] = model_path.name
    token_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(10)]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError("This constrained reference requires single-token decimal digits")
    ids = torch.tensor([item[0] for item in token_ids], device="cuda")
    prompts, features = {}, {}
    projection_errors = {str(count): {"max_ulp": 0.0, "max_absolute_error": 0.0} for count in (2, 10)}
    seen = set()
    for prompt in _study_partition_prompts(manifest, "train", "validation"):
        if prompt in seen:
            continue
        seen.add(prompt)
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model.model(**encoded, use_cache=False)
            hidden = outputs.last_hidden_state[0, -1]
            full_logits = model(**encoded, use_cache=False).logits[0, -1]
            for count in (2, 10):
                cached_logits = torch.nn.functional.linear(hidden.to(torch.bfloat16), model.lm_head.weight[ids[:count]])
                ulp_error, absolute_error = bf16_logit_error(full_logits[ids[:count]], cached_logits)
                projection_errors[str(count)]["max_ulp"] = max(projection_errors[str(count)]["max_ulp"], ulp_error)
                projection_errors[str(count)]["max_absolute_error"] = max(
                    projection_errors[str(count)]["max_absolute_error"], absolute_error
                )
        key = f"p{len(features):03d}"
        prompts[key] = prompt
        features[key] = hidden.float().cpu().contiguous()
    if any(item["max_ulp"] > 1.0 for item in projection_errors.values()):
        raise RuntimeError(f"Cached feature logits exceed one BF16 ULP: {projection_errors}")
    save_file(features, str(output / "features.safetensors"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    preparation = {
        "prompt_count": len(prompts),
        "cached_bf16_projection_errors": projection_errors,
        "max_cached_bf16_logit_error": max(item["max_absolute_error"] for item in projection_errors.values()),
        "max_cached_bf16_logit_ulp": max(item["max_ulp"] for item in projection_errors.values()),
        "bf16_ulp_floor": STUDY_BF16_ULP_FLOOR,
        "max_bf16_ulp": 1.0,
        "model_load_seconds": model_load_seconds,
        "preparation_seconds": time.perf_counter() - started,
        "manifest_sha256": _sha256(output / "manifest.json"),
        "features_sha256": _sha256(output / "features.safetensors"),
        "prompts_sha256": _sha256(output / "prompts.json"),
        "model_sha256": _model_hashes(model_path),
        "source_sha256": _source_hashes(),
        "model_config": {
            "model_type": model.config.model_type,
            "hidden_size": model.config.hidden_size,
            "vocab_size": model.config.vocab_size,
            "tie_word_embeddings": model.config.tie_word_embeddings,
        },
        "token_ids": [int(item[0]) for item in token_ids],
        "trainable_parameters": 10 * model.config.hidden_size,
        "runtime": _runtime_identity(),
    }
    (output / "preparation.json").write_text(
        json.dumps(preparation, indent=2),
        encoding="utf-8",
    )


def _load_study(model_path, study_dir):
    manifest = json.loads((study_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest != {**study_manifest(), "model_revision": model_path.name}:
        raise ValueError("Study manifest does not match the canonical frozen split and model revision")
    preparation = json.loads((study_dir / "preparation.json").read_text(encoding="utf-8"))
    for key, filename in (
        ("manifest_sha256", "manifest.json"),
        ("features_sha256", "features.safetensors"),
        ("prompts_sha256", "prompts.json"),
    ):
        if preparation.get(key) != _sha256(study_dir / filename):
            raise ValueError(f"Prepared study input hash mismatch: {filename}")
    if preparation.get("model_sha256") != _model_hashes(model_path):
        raise ValueError("Cached model configuration, weights, or tokenizer assets changed after preparation")
    if preparation.get("source_sha256") != _source_hashes():
        raise ValueError("Study source changed after preparation")
    prompts = json.loads((study_dir / "prompts.json").read_text(encoding="utf-8"))
    features = load_file(str(study_dir / "features.safetensors"), device="cuda")
    expected_prompts = set(_study_partition_prompts(manifest, "train", "validation"))
    if (
        set(features) != set(prompts)
        or set(prompts.values()) != expected_prompts
        or preparation.get("prompt_count") != len(prompts)
    ):
        raise ValueError("Prepared prompt and cached-feature sets do not match")
    return manifest, prompts, features, preparation


def _prompt_keys(prompts):
    by_text = {value: key for key, value in prompts.items()}
    return by_text


def _evaluate_study(model, rows, pairs, by_text, features):
    per_pair = []
    with torch.inference_mode():
        for a, b in pairs:
            first, answers = _prompts(a, b)
            h = features[by_text[first]].to(model.lm_head.weight.dtype)
            tool_probs = torch.softmax(torch.nn.functional.linear(h, rows[:2]).float(), -1)
            reward = success = 0.0
            for choice, answer_prompt in enumerate(answers):
                ah = features[by_text[answer_prompt]].to(model.lm_head.weight.dtype)
                answer_probs = torch.softmax(torch.nn.functional.linear(ah, rows).float(), -1)
                correct_prob = float(answer_probs[(a + b) % 10])
                success += float(tool_probs[choice]) * correct_prob
                reward += float(tool_probs[choice]) * (correct_prob + 0.25 * (choice == 0))
            per_pair.append({"pair": [a, b], "expected_reward": reward, "answer_success": success})
    return {
        "expected_reward": sum(item["expected_reward"] for item in per_pair) / len(per_pair),
        "answer_success": sum(item["answer_success"] for item in per_pair) / len(per_pair),
        "per_pair": per_pair,
    }


def _verify_cached_bf16_logits(model, tokenizer, prompts, by_text, features, ids):
    projection_errors = {str(count): {"max_ulp": 0.0, "max_absolute_error": 0.0} for count in (2, 10)}
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            actual = model(**encoded, use_cache=False).logits[0, -1]
            hidden = features[by_text[prompt]].to(torch.bfloat16)
            for count in (2, 10):
                rows = model.lm_head.weight[ids[:count]]
                cached = torch.nn.functional.linear(hidden, rows)
                ulp, absolute = bf16_logit_error(actual[ids[:count]], cached)
                projection_errors[str(count)]["max_ulp"] = max(projection_errors[str(count)]["max_ulp"], ulp)
                projection_errors[str(count)]["max_absolute_error"] = max(
                    projection_errors[str(count)]["max_absolute_error"], absolute
                )
    if any(item["max_ulp"] > 1.0 for item in projection_errors.values()):
        raise RuntimeError(f"Cached features exceed one BF16 ULP at constrained rows: {projection_errors}")
    return {
        "max_ulp": max(item["max_ulp"] for item in projection_errors.values()),
        "max_absolute_error": max(item["max_absolute_error"] for item in projection_errors.values()),
        "projections": projection_errors,
        "tolerance_ulp": 1.0,
    }


def run_study(model_path: Path, study_dir: Path):
    input_load_started = time.perf_counter()
    manifest, prompts, features, preparation = _load_study(model_path, study_dir)
    input_load_seconds = time.perf_counter() - input_load_started
    by_text = _prompt_keys(prompts)
    order = study_run_order()
    run_order_path = study_dir / "run_order.json"
    with run_order_path.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "config": study_config(),
                "order": order,
                "manifest_sha256": _sha256(study_dir / "manifest.json"),
                "source_sha256": preparation["source_sha256"],
                "model_sha256": preparation["model_sha256"],
            },
            handle,
            indent=2,
        )
    model_load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            use_safetensors=True,
        )
        .to("cuda")
        .eval()
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_started
    model.requires_grad_(False)
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone(), requires_grad=False)
    model.config.tie_word_embeddings = False
    token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
    ids = torch.tensor(token_ids, device="cuda")
    if token_ids != preparation["token_ids"]:
        raise ValueError("Tokenizer digit IDs changed after feature preparation")
    initial = model.lm_head.weight[ids].float().clone()
    root = study_dir / "runs"
    root.mkdir(exist_ok=True)

    async def do_run(arm, seed, order_index):
        run_dir = root / f"{arm}-seed-{seed}"
        run_dir.mkdir(exist_ok=False)
        head = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.SGD([head], lr=0.005)
        fence = StalenessFence(max_age=1)
        rngs, totals = {}, {
            "generated": 0,
            "accepted": 0,
            "stale": 0,
            "tail": 0,
            "action_tokens": 0,
            "prompt_tokens": 0,
            "mixed": 0,
        }
        timings = {
            "study_input_load_seconds": input_load_seconds,
            "model_load_seconds": model_load_seconds,
            "training_seconds": 0.0,
            "validation_seconds": 0.0,
            "consistency_seconds": 0.0,
            "checkpoint_seconds": 0.0,
        }
        versions_by_uid, prompt_tokens_by_uid, action_tokens_by_uid, terminal_records = {}, {}, {}, {}
        loss_curve, validation, checkpoint_receipts, consistency_checks = [], [], {}, {}

        def sample(prompt, count, uid):
            if uid not in rngs:
                rngs[uid] = torch.Generator(device="cuda").manual_seed(group_seed(seed, uid))
            hidden = features[by_text[prompt]].to("cuda")
            with torch.no_grad():
                hidden_bf16 = hidden.to(torch.bfloat16)
                rows_bf16 = model.lm_head.weight[ids[:count]]
                logits = torch.nn.functional.linear(hidden_bf16, rows_bf16).float()
                dist = torch.distributions.Categorical(logits=logits)
                index = torch.multinomial(dist.probs, 1, generator=rngs[uid])[0]
            return Action(hidden, int(index), float(dist.log_prob(index)), version)

        def start_group(index):
            uid = f"{index:04d}"
            a, b = schedule[index % len(schedule)]
            first, _ = _prompts(a, b)
            actions = [sample(first, 2, uid) for _ in range(2)]
            prompt_count = sum(len(tokenizer.encode(first, add_special_tokens=False)) for _ in actions)
            totals["generated"] += 1
            totals["prompt_tokens"] += prompt_count
            record_action_tokens(totals, action_tokens_by_uid, uid, 2)
            prompt_tokens_by_uid[uid] = prompt_count
            versions_by_uid[uid] = [action.version for action in actions]
            return uid, a, b, actions

        async def complete(uid, a, b, actions):
            full_actions, rewards = [], []
            for action in actions:
                _, answers = _prompts(a, b)
                answer_prompt = answers[action.index]
                answer = sample(answer_prompt, 10, uid)
                reward = float(answer.index == (a + b) % 10) + 0.25 * (action.index == 0)
                versions_by_uid[uid].append(answer.version)
                action_count = len(tokenizer.encode(answer_prompt, add_special_tokens=False))
                totals["prompt_tokens"] += action_count
                prompt_tokens_by_uid[uid] += action_count
                record_action_tokens(totals, action_tokens_by_uid, uid, 1)
                full_actions.extend([action, answer])
                rewards.append(reward)
                await asyncio.sleep(0)
            return Group(uid, min(versions_by_uid[uid]), full_actions, rewards)

        def apply(batch):
            nonlocal version
            for group in batch:
                totals["accepted"] += 1
                versions = sorted(set(versions_by_uid[group.uid]))
                totals["mixed"] += len(versions) > 1
            loss, _ = apply_policy_update(batch, head, optimizer, model, ids, training_dtype=torch.bfloat16)
            version += 1
            loss_curve.append(loss)

        def group_for(index):
            uid, a, b, first_actions = start_group(index)
            actions, rewards = [], []
            for action in first_actions:
                answer_prompt = _prompts(a, b)[1][action.index]
                answer = sample(answer_prompt, 10, uid)
                versions_by_uid[uid].append(answer.version)
                action_count = len(tokenizer.encode(answer_prompt, add_special_tokens=False))
                totals["prompt_tokens"] += action_count
                prompt_tokens_by_uid[uid] += action_count
                actions.extend([action, answer])
                rewards.append(float(answer.index == (a + b) % 10) + 0.25 * (action.index == 0))
                record_action_tokens(totals, action_tokens_by_uid, uid, 1)
            return Group(uid, min(versions_by_uid[uid]), actions, rewards)

        schedule = study_schedule(seed)
        version = 0

        def sync_time_start():
            torch.cuda.synchronize()
            return time.perf_counter()

        def sync_time_end(start):
            torch.cuda.synchronize()
            return time.perf_counter() - start

        def admitted(groups):
            batch = []
            for group in groups:
                decision = fence.decide(group.uid, group.oldest, version)
                terminal_records[group.uid] = {
                    "decision": decision.value,
                    "oldest": group.oldest,
                    "consumer_version": version,
                    "versions": list(versions_by_uid[group.uid]),
                }
                if decision == GroupDecision.KEEP:
                    batch.append(group)
                elif decision == GroupDecision.STALE:
                    totals["stale"] += 1
            if len(batch) != 2:
                raise RuntimeError(f"Expected two accepted groups at version {version}, got {len(batch)}")
            return batch

        async def async_groups(items):
            ready = asyncio.Event()
            resume = asyncio.Event()

            async def carry(item):
                uid, a, b, first_actions = item
                ready.set()
                await resume.wait()
                return await complete(uid, a, b, first_actions)

            tasks = [asyncio.create_task(carry(item)) for item in items]
            await ready.wait()
            return tasks, resume

        with torch.no_grad():
            model.lm_head.weight.index_copy_(0, ids, initial.to(model.lm_head.weight.dtype))
        tick = sync_time_start()
        consistency_checks["initial"] = _verify_cached_bf16_logits(
            model, tokenizer, list(by_text), by_text, features, ids
        )
        timings["consistency_seconds"] += sync_time_end(tick)
        tick = sync_time_start()
        initial_validation = _evaluate_study(
            model,
            model.lm_head.weight[ids],
            manifest["partitions"]["validation"],
            by_text,
            features,
        )
        timings["validation_seconds"] += sync_time_end(tick)
        pending_groups = None
        for step in range(STUDY_UPDATES):
            started = sync_time_start()
            if arm == "sync" or step == 0:
                groups = [group_for(step * STUDY_GROUPS_PER_UPDATE + j) for j in range(STUDY_GROUPS_PER_UPDATE)]
            else:
                assert pending_groups is not None, "Async carry must complete before its next update"
                groups = pending_groups
            if arm == "async":
                partials = [
                    start_group((step + 1) * STUDY_GROUPS_PER_UPDATE + j) for j in range(STUDY_GROUPS_PER_UPDATE)
                ]
                pending_tasks, pending_resume = await async_groups(partials)
            batch = admitted(groups)
            apply(batch)
            timings["training_seconds"] += sync_time_end(started)
            if arm == "async" and step < STUDY_UPDATES - 1:
                carry_started = sync_time_start()
                pending_resume.set()
                pending_groups = await asyncio.gather(*pending_tasks)
                timings["training_seconds"] += sync_time_end(carry_started)
            elif arm == "async" and step == STUDY_UPDATES - 1:
                carry_started = sync_time_start()
                pending_resume.set()
                tail = await asyncio.gather(*pending_tasks)
                timings["training_seconds"] += sync_time_end(carry_started)
                totals["tail"] += len(tail)
                for group in tail:
                    terminal_records[group.uid] = {
                        "decision": "tail",
                        "oldest": group.oldest,
                        "consumer_version": version,
                        "versions": list(versions_by_uid[group.uid]),
                    }
            if version in STUDY_CHECKPOINTS:
                tick = sync_time_start()
                consistency_checks[str(version)] = _verify_cached_bf16_logits(
                    model,
                    tokenizer,
                    list(by_text),
                    by_text,
                    features,
                    ids,
                )
                timings["consistency_seconds"] += sync_time_end(tick)
                tick = sync_time_start()
                val = _evaluate_study(
                    model, model.lm_head.weight[ids], manifest["partitions"]["validation"], by_text, features
                )
                timings["validation_seconds"] += sync_time_end(tick)
                validation.append({"step": version, **val})
                tick = sync_time_start()
                filename = f"step-{version}.safetensors"
                master = head.detach().cpu().clone()
                installed = model.lm_head.weight[ids].detach().cpu().clone()
                if not torch.isfinite(master).all() or not torch.isfinite(installed).all():
                    raise RuntimeError(f"Nonfinite checkpoint head at update {version}")
                save_file(
                    {
                        "head_weights": master,
                        "installed_bf16_rows": installed,
                        "initial_head": initial.cpu(),
                        "token_ids": ids.cpu(),
                    },
                    str(run_dir / filename),
                )
                restored = load_file(str(run_dir / filename), device="cpu")
                if (
                    not torch.equal(restored["head_weights"], master)
                    or not torch.equal(restored["installed_bf16_rows"], master.to(torch.bfloat16))
                    or not torch.equal(restored["initial_head"], initial.cpu())
                    or not torch.equal(restored["token_ids"], ids.cpu())
                ):
                    raise RuntimeError(f"Native checkpoint reload mismatch at update {version}")
                checkpoint_receipts[str(version)] = {
                    "filename": filename,
                    "sha256": _sha256(run_dir / filename),
                    "head_sha256": _tensor_sha256(restored["head_weights"]),
                    "installed_bf16_sha256": _tensor_sha256(restored["installed_bf16_rows"]),
                    "initial_head_sha256": _tensor_sha256(restored["initial_head"]),
                }
                timings["checkpoint_seconds"] += sync_time_end(tick)
        metrics = {
            "run_id": f"{arm}-seed-{seed}",
            "run_order_index": order_index,
            "arm": arm,
            "seed": seed,
            "updates": version,
            "counts": totals,
            "validation": validation,
            "losses": loss_curve,
            "timings": timings,
            "initial_validation": initial_validation,
            "group_versions": versions_by_uid,
            "prompt_tokens_by_uid": prompt_tokens_by_uid,
            "action_tokens_by_uid": action_tokens_by_uid,
            "terminal_records": terminal_records,
            "checkpoint_receipts": checkpoint_receipts,
            "consistency_checks": consistency_checks,
            "config": study_config(),
            "manifest_sha256": _sha256(study_dir / "manifest.json"),
            "data_sha256": {
                "manifest": _sha256(study_dir / "manifest.json"),
                "features": preparation["features_sha256"],
                "prompts": preparation["prompts_sha256"],
                "preparation": _sha256(study_dir / "preparation.json"),
            },
            "schedule_sha256": hashlib.sha256(
                json.dumps(study_schedule(seed), separators=(",", ":")).encode()
            ).hexdigest(),
            "model_sha256": preparation["model_sha256"],
            "source_sha256": preparation["source_sha256"],
            "runtime": _runtime_identity(),
            "model_config": preparation["model_config"],
            "trainable_parameters": 10 * model.config.hidden_size,
            "training_policy": "BF16 cached-feature sampling, FP32 master head with differentiable BF16 training forward, installed BF16 model rows",
            "runtime_policy": "one asyncio event loop; CUDA kernels serialized",
        }
        validate_study_metrics(metrics, arm, seed, order_index)
        metrics["run_passed"] = True
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        run_receipt = {
            "run_id": metrics["run_id"],
            "run_order_index": order_index,
            "arm": arm,
            "seed": seed,
            "updates": version,
            "counts": totals,
            "schedule_sha256": metrics["schedule_sha256"],
            "data_sha256": metrics["data_sha256"],
            "runtime": metrics["runtime"],
            "model_config": metrics["model_config"],
            "trainable_parameters": metrics["trainable_parameters"],
            "training_policy": metrics["training_policy"],
            "runtime_policy": metrics["runtime_policy"],
            "checkpoint_receipts": checkpoint_receipts,
            "manifest_sha256": metrics["manifest_sha256"],
            "source_sha256": metrics["source_sha256"],
            "model_sha256": metrics["model_sha256"],
            "config": metrics["config"],
            "metrics_sha256": _sha256(metrics_path),
            "run_passed": True,
        }
        (run_dir / "run_receipt.json").write_text(json.dumps(run_receipt, indent=2), encoding="utf-8")

    for entry in order:
        asyncio.run(do_run(entry["arm"], entry["seed"], entry["index"]))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _build_selection(study_dir: Path, model_path: Path):
    manifest_path, preparation_path = study_dir / "manifest.json", study_dir / "preparation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if manifest != {**study_manifest(), "model_revision": model_path.name}:
        raise ValueError("Selection manifest does not match the frozen study split")
    source_sha256, model_sha256 = _source_hashes(), _model_hashes(model_path)
    if preparation.get("source_sha256") != source_sha256 or preparation.get("model_sha256") != model_sha256:
        raise ValueError("Current source or cached model assets differ from preparation")
    for key, filename in (
        ("manifest_sha256", "manifest.json"),
        ("features_sha256", "features.safetensors"),
        ("prompts_sha256", "prompts.json"),
    ):
        if preparation.get(key) != _sha256(study_dir / filename):
            raise ValueError(f"Prepared study input hash mismatch: {filename}")
    manifest_sha256 = _sha256(manifest_path)
    data_sha256 = {
        "manifest": manifest_sha256,
        "features": _sha256(study_dir / "features.safetensors"),
        "prompts": _sha256(study_dir / "prompts.json"),
        "preparation": _sha256(preparation_path),
    }
    expected_order = study_run_order()
    order_receipt = json.loads((study_dir / "run_order.json").read_text(encoding="utf-8"))
    if order_receipt != {
        "config": study_config(),
        "order": expected_order,
        "manifest_sha256": manifest_sha256,
        "source_sha256": source_sha256,
        "model_sha256": model_sha256,
    }:
        raise ValueError("Run order receipt does not match required interleaved schedule and inputs")
    run_root = study_dir / "runs"
    _validate_run_inventory(run_root, expected_order)
    selections, candidates, run_receipts, paired_deltas = {}, {}, {}, {}
    initial_head_sha256 = None
    for entry in expected_order:
        arm, seed = entry["arm"], entry["seed"]
        run_id = f"{arm}-seed-{seed}"
        run_dir = run_root / run_id
        expected_files = {
            "metrics.json",
            "run_receipt.json",
            *(f"step-{step}.safetensors" for step in STUDY_CHECKPOINTS),
        }
        if {path.name for path in run_dir.iterdir() if path.is_file()} != expected_files:
            raise ValueError(f"Run files do not match the complete metrics/checkpoint inventory: {run_id}")
        metrics_path, receipt_path = run_dir / "metrics.json", run_dir / "run_receipt.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if run_receipt.get("metrics_sha256") != _sha256(metrics_path):
            raise ValueError(f"Metrics hash mismatch: {run_id}")
        validate_selection_run(metrics, run_receipt, entry, manifest_sha256, source_sha256, model_sha256, data_sha256)
        if metrics.get("runtime") != preparation.get("runtime") or metrics.get("model_config") != preparation.get(
            "model_config"
        ):
            raise ValueError(f"Runtime/model configuration differs from preparation: {run_id}")
        if metrics.get("trainable_parameters") != preparation.get("trainable_parameters"):
            raise ValueError(f"Trainable parameter count differs from preparation: {run_id}")
        run_receipts[run_id] = {"metrics_sha256": _sha256(metrics_path), "run_receipt_sha256": _sha256(receipt_path)}
        initial = metrics["initial_validation"]
        for item in metrics["validation"]:
            step = item["step"]
            cp_receipt = metrics["checkpoint_receipts"][str(step)]
            cp_path = run_dir / cp_receipt["filename"]
            if _sha256(cp_path) != cp_receipt["sha256"]:
                raise ValueError(f"Checkpoint file hash mismatch: {run_id} step {step}")
            tensors = load_file(str(cp_path), device="cpu")
            if set(tensors) != {"head_weights", "installed_bf16_rows", "initial_head", "token_ids"}:
                raise ValueError(f"Checkpoint tensor inventory mismatch: {run_id} step {step}")
            head, installed = tensors["head_weights"], tensors["installed_bf16_rows"]
            if (
                not torch.isfinite(head).all()
                or not torch.isfinite(installed).all()
                or not torch.equal(installed, head.to(torch.bfloat16))
            ):
                raise ValueError(f"Checkpoint BF16 rows do not exactly reload from FP32 master: {run_id} step {step}")
            if (
                _tensor_sha256(head) != cp_receipt["head_sha256"]
                or _tensor_sha256(installed) != cp_receipt["installed_bf16_sha256"]
            ):
                raise ValueError(f"Checkpoint tensor hash mismatch: {run_id} step {step}")
            if tensors["token_ids"].tolist() != preparation["token_ids"]:
                raise ValueError(f"Checkpoint token IDs differ from preparation: {run_id} step {step}")
            if not torch.isfinite(tensors["initial_head"]).all():
                raise ValueError(f"Nonfinite checkpoint initial head: {run_id} step {step}")
            initial_hash = _tensor_sha256(tensors["initial_head"])
            if cp_receipt.get("initial_head_sha256") != initial_hash:
                raise ValueError(f"Checkpoint initial-head hash mismatch: {run_id} step {step}")
            if initial_head_sha256 is None:
                initial_head_sha256 = initial_hash
            elif initial_head_sha256 != initial_hash:
                raise ValueError("Paired runs did not start from identical output rows")
            candidates[f"{run_id}@{step}"] = {
                "sha256": cp_receipt["sha256"],
                "head_sha256": cp_receipt["head_sha256"],
                "installed_bf16_sha256": cp_receipt["installed_bf16_sha256"],
                "initial_head_sha256": initial_hash,
            }
        best = max(metrics["validation"], key=lambda item: (item["expected_reward"], -item["step"]))
        selections[run_id] = {
            "arm": arm,
            "seed": seed,
            "step": best["step"],
            "initial_validation": initial,
            "selected_validation": best,
            "validation_expected_reward": best["expected_reward"],
            "validation_answer_success": best["answer_success"],
            "validation_expected_reward_improvement": best["expected_reward"] - initial["expected_reward"],
            "validation_answer_success_improvement": best["answer_success"] - initial["answer_success"],
            "checkpoint": str((run_dir / f"step-{best['step']}.safetensors").relative_to(study_dir)),
            "checkpoint_sha256": metrics["checkpoint_receipts"][str(best["step"])]["sha256"],
        }
    for seed in STUDY_SEEDS:
        sync, async_result = selections[f"sync-seed-{seed}"], selections[f"async-seed-{seed}"]
        paired_deltas[str(seed)] = {
            "expected_reward": async_result["selected_validation"]["expected_reward"]
            - sync["selected_validation"]["expected_reward"],
            "answer_success": async_result["selected_validation"]["answer_success"]
            - sync["selected_validation"]["answer_success"],
            "validation_steps": {"sync": sync["step"], "async": async_result["step"]},
        }
    return {
        "schema": 2,
        "manifest_sha256": manifest_sha256,
        "features_sha256": _sha256(study_dir / "features.safetensors"),
        "prompts_sha256": _sha256(study_dir / "prompts.json"),
        "preparation_sha256": _sha256(preparation_path),
        "source_sha256": source_sha256,
        "model_sha256": model_sha256,
        "config": study_config(),
        "run_order": expected_order,
        "run_receipts": run_receipts,
        "candidate_checkpoints": candidates,
        "selections": selections,
        "paired_validation_deltas": paired_deltas,
        "test_accessed": False,
    }


def select_study(model_path: Path, study_dir: Path):
    selection_path = study_dir / "selection.json"
    if selection_path.exists():
        raise FileExistsError("Refusing to overwrite an existing selection receipt")
    receipt = _build_selection(study_dir, model_path)
    with selection_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)


def test_study(model_path: Path, study_dir: Path):
    output_path = study_dir / "test_receipt.json"
    if output_path.exists():
        raise FileExistsError("Refusing to overwrite an existing final-test receipt")
    selection_verification_started = time.perf_counter()
    receipt_path = study_dir / "selection.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    frozen_selection = _build_selection(study_dir, model_path)
    if receipt != frozen_selection or receipt.get("test_accessed"):
        raise ValueError("Selection receipt or candidate hashes differ from deterministic frozen selection")
    selection_verification_seconds = time.perf_counter() - selection_verification_started
    input_load_started = time.perf_counter()
    manifest, prompts, features, preparation = _load_study(model_path, study_dir)
    input_load_seconds = time.perf_counter() - input_load_started
    model_load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if preparation["token_ids"] != [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)]:
        raise ValueError("Tokenizer digit IDs changed before final test")
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            use_safetensors=True,
        )
        .to("cuda")
        .eval()
    )
    model.requires_grad_(False)
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone(), requires_grad=False)
    model.config.tie_word_embeddings = False
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_started
    token_ids = torch.tensor([tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)], device="cuda")
    feature_preparation_started = time.perf_counter()
    test_prompts = _study_partition_prompts(manifest, "test")
    max_logit_error = {"initial": None, "selected": {}}
    for prompt in test_prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            hidden = model.model(**encoded, use_cache=False).last_hidden_state[0, -1]
        key = f"p{len(features):03d}"
        prompts[key] = prompt
        features[key] = hidden.float().contiguous()
    feature_preparation_seconds = time.perf_counter() - feature_preparation_started
    by_text = _prompt_keys(prompts)
    initial_rows = model.lm_head.weight[token_ids]
    consistency_started = time.perf_counter()
    max_logit_error["initial"] = _verify_cached_bf16_logits(
        model, tokenizer, test_prompts, by_text, features, token_ids
    )
    consistency_seconds = time.perf_counter() - consistency_started
    evaluation_started = time.perf_counter()
    initial_result = _evaluate_study(model, initial_rows, manifest["partitions"]["test"], by_text, features)
    initial_evaluation_seconds = time.perf_counter() - evaluation_started
    results = {}
    checkpoint_load_seconds = 0.0
    for key, selection in receipt["selections"].items():
        checkpoint_load_started = time.perf_counter()
        saved = load_file(str(study_dir / selection["checkpoint"]), device="cuda")
        checkpoint_load_seconds += time.perf_counter() - checkpoint_load_started
        with torch.no_grad():
            model.lm_head.weight.index_copy_(0, token_ids, saved["installed_bf16_rows"])
        consistency_started = time.perf_counter()
        max_logit_error["selected"][key] = _verify_cached_bf16_logits(
            model, tokenizer, test_prompts, by_text, features, token_ids
        )
        consistency_seconds += time.perf_counter() - consistency_started
        selected_evaluation_started = time.perf_counter()
        results[key] = _evaluate_study(
            model,
            model.lm_head.weight[token_ids],
            manifest["partitions"]["test"],
            by_text,
            features,
        )
        initial_evaluation_seconds += time.perf_counter() - selected_evaluation_started
    paired_deltas = {}
    for seed in STUDY_SEEDS:
        paired_deltas[str(seed)] = {
            "from_initial": {
                arm: {
                    "expected_reward": results[f"{arm}-seed-{seed}"]["expected_reward"]
                    - initial_result["expected_reward"],
                    "answer_success": results[f"{arm}-seed-{seed}"]["answer_success"]
                    - initial_result["answer_success"],
                }
                for arm in ("sync", "async")
            },
            "async_minus_sync": {
                "expected_reward": results[f"async-seed-{seed}"]["expected_reward"]
                - results[f"sync-seed-{seed}"]["expected_reward"],
                "answer_success": results[f"async-seed-{seed}"]["answer_success"]
                - results[f"sync-seed-{seed}"]["answer_success"],
            },
        }
    output = {
        "schema": 2,
        "selection_sha256": _sha256(receipt_path),
        "candidate_checkpoints": receipt["candidate_checkpoints"],
        "initial_test": initial_result,
        "test_results": results,
        "paired_test_deltas_from_initial": paired_deltas,
        "test_pairs": manifest["partitions"]["test"],
        "quality_is_reported_not_a_pass_gate": True,
        "cached_logit_consistency": max_logit_error,
        "selection_verification_seconds": selection_verification_seconds,
        "study_input_load_seconds": input_load_seconds,
        "model_load_seconds": model_load_seconds,
        "test_feature_preparation_seconds": feature_preparation_seconds,
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "consistency_seconds": consistency_seconds,
        "evaluation_seconds": initial_evaluation_seconds,
    }
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)


@dataclass
class Action:
    hidden: torch.Tensor
    index: int
    logprob: float
    version: int


@dataclass
class Group:
    uid: str
    oldest: int
    actions: list[Action]
    rewards: list[float]


def tool(a, b, choice):
    if choice not in (0, 1):
        raise ValueError("Only owned sum and difference tools are available")
    return (a + b) % 10 if choice == 0 else abs(a - b)


async def train(model_path: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(15)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            use_safetensors=True,
        )
        .to("cuda")
        .eval()
    )
    model.requires_grad_(False)
    # Untie the head so head updates cannot silently change the frozen input embeddings.
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone(), requires_grad=False)
    model.config.tie_word_embeddings = False
    token_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(10)]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError("This constrained reference requires single-token decimal digits")
    ids = torch.tensor([item[0] for item in token_ids], device="cuda")
    initial = model.lm_head.weight[ids].float().clone()
    head = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.SGD([head], lr=0.005)
    version = 0
    fence = StalenessFence(max_age=1)
    queue = asyncio.Queue()
    tasks = []
    terminal = {}
    trained_uids = []
    curve = []
    losses = []
    mixed = []
    events_path = output / "events.jsonl"
    events = events_path.open("w", encoding="utf-8")

    def emit(kind, **data):
        events.write(json.dumps({"event": kind, "version": version, **data}) + "\n")
        events.flush()

    hidden_cache = {}

    def hidden(prompt):
        if prompt not in hidden_cache:
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                result = model.model(**encoded, use_cache=False).last_hidden_state[0, -1].float().clone()
            hidden_cache[prompt] = result.clone()
        return hidden_cache[prompt]

    def first_prompt(a, b):
        return f"Compute ({a}+{b}) modulo 10. Choose tool 0 for sum, 1 for difference. Tool:"

    def second_prompt(a, b, choice):
        return first_prompt(a, b) + f" {choice}\nTool result: {tool(a, b, choice)}\nAnswer digit:"

    def sample(prompt, count):
        h = hidden(prompt)
        with torch.no_grad():
            dist = torch.distributions.Categorical(logits=torch.nn.functional.linear(h, head[:count]))
            index = dist.sample()
        return Action(h, int(index), float(dist.log_prob(index)), version)

    async def rollout(uid, a, b, gate, ready):
        oldest = version
        first = [sample(first_prompt(a, b), 2) for _ in range(2)]
        emit("partial", uid=uid, oldest=oldest, tool_actions=[a.index for a in first])
        ready.set()
        await gate.wait()
        actions, rewards = [], []
        for action in first:
            observation = tool(a, b, action.index)
            answer = sample(second_prompt(a, b, action.index), 10)
            reward = float(answer.index == (a + b) % 10) + 0.25 * (action.index == 0)
            actions.extend([action, answer])
            rewards.append(reward)
            emit(
                "tool_turn",
                uid=uid,
                tool=action.index,
                observation=observation,
                answer=answer.index,
                reward=reward,
                policy_versions=[action.version, answer.version],
            )
            await asyncio.sleep(0)
        await queue.put(Group(uid, oldest, actions, rewards))
        emit("completed", uid=uid, oldest=oldest)

    async def launch(uid, a, b):
        gate, ready = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(rollout(uid, a, b, gate, ready))
        tasks.append(task)
        task.add_done_callback(lambda _: ready.set())
        await ready.wait()
        if task.done():
            await task
        return gate, task

    def evaluate():
        expected = []
        with torch.no_grad():
            for a, b in [(6, 7), (7, 8), (8, 9), (9, 6)]:
                tool_probs = torch.softmax(torch.nn.functional.linear(hidden(first_prompt(a, b)), head[:2]), -1)
                total = 0.0
                for choice in (0, 1):
                    answers = torch.softmax(torch.nn.functional.linear(hidden(second_prompt(a, b, choice)), head), -1)
                    total += float(tool_probs[choice] * (answers[(a + b) % 10] + 0.25 * (choice == 0)))
                expected.append(total)
        curve.append({"version": version, "heldout_expected_reward": sum(expected) / len(expected)})

    try:
        emit("loaded", model=str(model_path), trainable_parameters=head.numel(), backbone_frozen=True)
        evaluate()
        slow_gate, slow_task = await launch("straggler", 1, 2)
        carry = None
        for step in range(4):
            fast_count = 2 if carry is None else 1
            for j in range(fast_count):
                gate, task = await launch(f"fast-{step}-{j}", step + 1, j + 2)
                gate.set()
                await task
            next_gate, next_task = await launch(f"carry-{step}", step + 2, 3)
            batch = []
            while len(batch) < 2:
                group = await asyncio.wait_for(queue.get(), timeout=30)
                decision = fence.decide(group.uid, group.oldest, version)
                emit("fence", uid=group.uid, oldest=group.oldest, decision=decision.value)
                if decision == GroupDecision.DUPLICATE:
                    continue
                terminal[group.uid] = decision.value
                if decision == GroupDecision.STALE:
                    continue
                batch.append(group)
            emit("pause", in_flight=["straggler", f"carry-{step}"] if step < 2 else [f"carry-{step}"])
            for group in batch:
                trained_uids.append(group.uid)
                versions = sorted({a.version for a in group.actions})
                if len(versions) > 1:
                    mixed.append({"uid": group.uid, "versions": versions})
            loss, grad_norm = apply_policy_update(batch, head, optimizer, model, ids)
            version += 1
            losses.append(loss)
            emit("update", loss=losses[-1], gradient_norm=grad_norm, trained=[g.uid for g in batch])
            evaluate()
            emit("resume", in_flight=f"carry-{step}")
            next_gate.set()
            await next_task
            carry = next_task
            if version == 2:
                slow_gate.set()
                await slow_task
        while not queue.empty():
            group = queue.get_nowait()
            terminal[group.uid] = "epoch_tail_filtered"
            emit("epoch_tail_filtered", uid=group.uid)
        delta = head.detach() - initial
        delta_norm = float(delta.norm())
        actual_delta = float((model.lm_head.weight[ids].float() - initial).norm())
        assert delta_norm > 0 and actual_delta > 0
        assert terminal["straggler"] == "stale"
        assert mixed and len(trained_uids) == len(set(trained_uids)) == 8
        assert len(terminal) == 10
        checkpoint = {
            "version": version,
            "terminal_uids": terminal,
            "trained_uids": trained_uids,
            "model_revision": model_path.name,
            "token_ids": ids.tolist(),
        }
        (output / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        restored = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
        assert set(restored["terminal_uids"]) == set(terminal)
        assert not [uid for uid in terminal if uid not in restored["terminal_uids"]]
        save_file(
            {
                "head_delta": delta.cpu(),
                "token_ids": ids.cpu(),
                "initial_head": initial.cpu(),
                "head_weights": head.detach().cpu(),
            },
            str(output / "policy-delta.safetensors"),
        )
        saved = load_file(str(output / "policy-delta.safetensors"), device="cuda")
        restored_head = saved["head_weights"]
        assert torch.equal(restored_head, head.detach())
        with torch.no_grad():
            head.copy_(restored_head)
        candidates = list(terminal) + ["resumed-fresh"]
        pending = [uid for uid in candidates if uid not in restored["terminal_uids"]]
        assert pending == ["resumed-fresh"]
        gate, task = await launch(pending[0], 4, 5)
        gate.set()
        await task
        resumed = queue.get_nowait()
        assert resumed.uid == "resumed-fresh" and resumed.oldest == restored["version"]
        assert fence.decide(resumed.uid, resumed.oldest, version) == GroupDecision.KEEP
        metrics = {
            "updates": version,
            "losses": losses,
            "parameter_delta_l2": delta_norm,
            "actual_model_head_delta_l2": actual_delta,
            "mixed_version_groups": mixed,
            "heldout_curve": curve,
            "trained_groups": len(trained_uids),
            "terminal_groups": terminal,
            "seed": 15,
            "scope": "Qwen3-0.6B frozen backbone, ten pretrained output rows, constrained actions",
            "native_distributed_backend": "not executed",
        }
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        emit(
            "checkpoint_reload",
            skipped_terminal=len(restored["terminal_uids"]),
            fresh_uid=resumed.uid,
            head_equal=True,
            optimizer="SGD without momentum has no persistent tensor state",
        )
        print(json.dumps(metrics, indent=2))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        events.close()


def run(model_path, output):
    asyncio.run(train(model_path, output))
