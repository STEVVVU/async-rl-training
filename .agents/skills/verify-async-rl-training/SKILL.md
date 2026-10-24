---
name: verify-async-rl-training
description: Drive the cached pretrained local async RL reference and check strict staleness fencing. Use after trainer or local reference changes.
---

# Launch

Run from the repository root. Use an isolated uv runner and an explicit existing ML interpreter without installing into it.

```powershell
$uv = 'C:/Users/steve/.codex/tools/uv-0.13.4/uv-0.13.4.data/scripts/uv.exe'
$ml = 'C:/Users/steve/.codex/tools/coursework-ml/Scripts/python.exe'
& $uv run --isolated --no-project --python $ml -- cmd.exe /d /c $ml scripts/verify_async_rl.py doctor
```

The Windows `cmd.exe` invocation explicitly selects the existing ML interpreter. A Python script passed directly to `uv run --isolated --python $ml` runs in a new environment without the installed ML packages. This command does not isolate or modify those packages. Doctor verified the existing interpreter, Torch and CUDA through this invocation. No server is needed. Each drive starts and exits its own process. Coordinate exclusive GPU use with other jobs.

# Doctor

The command above reports Python, Torch, Transformers, GPU and pinned model cache readiness. It never downloads models or loads remote Python. Local requirements are recorded in `examples/local_async/requirements.txt`. This backend has a separate compatibility envelope from upstream SkyRL.

# Drive

```powershell
& $uv run --isolated --no-project --python $ml -- cmd.exe /d /c $ml scripts/verify_async_rl.py drive --output .evidence/my-new-run
```

Choose an unused output directory. See [the feature map](features/README.md). This is real pretrained local policy training, not the distributed FSDP/vLLM backend. The native fully async example uses `examples.train.fully_async.main_fully_async`. Its Linux GPU checks are unverified on Windows.

# Evidence

Run `& $uv run --isolated --no-project --python $ml -- cmd.exe /d /c $ml scripts/verify_async_rl.py evidence --output .evidence/my-new-run`. Preserve `events.jsonl`, `metrics.json`, `checkpoint.json`, and `policy-delta.safetensors`. The evidence command checks sampled group admission, version brackets, nonzero parameter changes, checkpoint reload, and heldout curve availability. No assertion depends on generated prose or an improvement claim from a tiny run.

# Cleanup

The drive must exit and release CUDA. Run `& $uv run --isolated --no-project --python $ml -- cmd.exe /d /c $ml scripts/verify_async_rl.py cleanup --output .evidence/my-new-run`. It rechecks retained evidence. Confirm `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` no longer shows the owned drive PID. Never remove evidence or kill unrelated processes. No persistent server or process is created.

Proven locally on October 21, 2026, through doctor, drive, evidence, and cleanup with `.evidence/final-reference`. The native distributed journey remains blocked. See [local scope and results](../../../README.md).

## Matched study

The study stages use the same cached model and explicit interpreter. Preparation requires a new study directory, at least 10 GiB free disk, 8 GiB available system memory, and 4 GiB free CUDA memory. The driver uses one process and asyncio tasks only. It does not spawn worker processes. Training and test evaluation repeat those capacity guards.

```powershell
& $uv run --isolated --no-project --python $ml -- cmd.exe /d /c $ml scripts/verify_async_rl.py study-prepare --study .evidence/matched-study
& $uv run --isolated --no-project --python $ml -- cmd.exe /d /c $ml scripts/verify_async_rl.py study-run --study .evidence/matched-study
& $uv run --isolated --no-project --python $ml -- cmd.exe /d /c $ml scripts/verify_async_rl.py study-select --study .evidence/matched-study
& $uv run --isolated --no-project --python $ml -- cmd.exe /d /c $ml scripts/verify_async_rl.py study-test --study .evidence/matched-study
```

Treat `study-test` as a separate final action after reviewing `selection.json`. Preparation caches only train and validation prompts; it does not read test prompts. The five-seed/two-arm run and validation selection do not access test prompts. Selection requires ten interleaved run receipts, all three validation checkpoints for each run, full metrics/checkpoint/source/model/config hashes, and exact native checkpoint reloads. It binds each run's manifest, feature, prompt, and preparation hashes to current prepared files. It selects by validation expected reward only and reports negative improvements. The test action recomputes the frozen selection and verifies every candidate checkpoint before reading test prompts; it refuses to overwrite a prior test receipt. Cached-feature logits are checked within one BF16 ULP separately for the constrained two-row and ten-row projections, with per-projection ULP and absolute error retained in receipts.

Launch/Doctor, preparation, ten-run Drive, validation selection, separate final test, retained-evidence audit and process-exit Cleanup passed for `.evidence/matched-study-final-20261022-104451` at source `059ae60ef128346701cb7f260da183f1c562e970`. All selected policies underperformed the initial head. The default four-update Drive/Evidence/Cleanup also passed again at `.evidence/reference-regression-20261022-1051`. Read [the completed results](../../../docs/matched-study-results-20261022.md) before reporting numbers. The native Linux distributed journey remains unverified.

## Resource-light head-quality comparison

Run on the existing ML interpreter from the repository root. `train` requires a prior matched-study feature cache and an explicit native checkpoint containing `initial_head` and `token_ids`. It runs five seeds × two groups per update × 256 updates for clipped (`0.5`) and unclipped arms on CPU with one Torch thread; it writes native safetensors checkpoints at updates 32, 128, and 256, run metrics, source/input hashes, and validation-only selection. Output must be a new directory.

```powershell
& $uv run --isolated --no-project --python $ml -- cmd.exe /d /c $ml scripts/verify_head_quality.py train --features-from .evidence/matched-study --initial-checkpoint .evidence/matched-study/runs/sync-seed-11/step-32.safetensors --output .evidence/head-quality
```

Review `selection.json` before the separate GPU test. The test verifies selection against hashed validation metrics and validates source, input, and selected checkpoint hashes before reading held-out prompts or loading the model. It evaluates the reused held-out pair set and cached-feature BF16 logits. This is CPU head training followed by GPU evaluation, not asynchronous/distributed training or full-model fine-tuning.

```powershell
& $uv run --isolated --no-project --python $ml -- cmd.exe /d /c $ml scripts/verify_head_quality.py test --features-from .evidence/matched-study --model "$env:USERPROFILE/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca" --output .evidence/head-quality
```
