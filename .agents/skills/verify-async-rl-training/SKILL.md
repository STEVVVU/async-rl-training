---
name: verify-async-rl-training
description: Drive the cached pretrained local async RL reference and check strict staleness fencing. Use after trainer or local reference changes.
---

# Launch

Run from the repository root. Use an isolated uv runner and an explicit existing ML interpreter without installing into it.

```powershell
$uv = 'C:/Users/steve/.codex/tools/uv-0.13.4/uv-0.13.4.data/scripts/uv.exe'
$ml = 'C:/Users/steve/.codex/tools/coursework-ml/Scripts/python.exe'
& $uv run --isolated --no-project $ml scripts/verify_async_rl.py doctor
```

No server is needed. Each drive starts and exits its own process. The doctor must find cached safetensors and CUDA before the reference run. Coordinate exclusive GPU use with other jobs.

# Doctor

The command above reports Python, Torch, Transformers, GPU and pinned model cache readiness. It never downloads models or loads remote Python. Local requirements are recorded in `examples/local_async/requirements.txt`. This backend has a separate compatibility envelope from upstream SkyRL.

# Drive

```powershell
& $uv run --isolated --no-project $ml scripts/verify_async_rl.py drive --output .evidence/my-new-run
```

Choose an unused output directory. See [the feature map](features/README.md). This is real pretrained local policy training, not the distributed FSDP/vLLM backend. The native fully async example uses `examples.train.fully_async.main_fully_async`. Its Linux GPU checks are unverified on Windows.

# Evidence

Run `& $uv run --isolated --no-project $ml scripts/verify_async_rl.py evidence --output .evidence/my-new-run`. Preserve `events.jsonl`, `metrics.json`, `checkpoint.json`, and `policy-delta.safetensors`. The evidence command checks sampled group admission, version brackets, nonzero parameter changes, checkpoint reload, and heldout curve availability. No assertion depends on generated prose or an improvement claim from a tiny run.

# Cleanup

The drive must exit and release CUDA. Run `& $uv run --isolated --no-project $ml scripts/verify_async_rl.py cleanup --output .evidence/my-new-run`. It rechecks retained evidence. Confirm `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` no longer shows the owned drive PID. Never remove evidence or kill unrelated processes. No persistent server or process is created.

Proven locally on October 21, 2026, through doctor, drive, evidence, and cleanup with `.evidence/final-reference`. The native distributed journey remains blocked. See [local scope and results](../../../README.local.md).
