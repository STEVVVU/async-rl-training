# Retained pretrained reference measurements

`metrics.json` and `checkpoint.json` preserve the measured values from the actual October 21, 2026 local CUDA run. They are historical evidence uploaded by CPU CI, not measurements produced by that job. The run used cached official Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, Python 3.11.9, Torch 2.11.0+cu128, Transformers 5.20.2, and an RTX 5080 Laptop GPU.

Four real policy updates changed the model head and admitted eight groups, including three mixed-version groups. The strict fence rejected the two-update-old straggler. Heldout expected reward fell from 0.760472 to 0.543493. This demonstrates training mechanics and does not establish quality improvement, convergence, throughput, or native distributed correctness.

The full local run also retains sampled tool events and safetensors head weights in `.evidence/final-reference/`. Those larger local artifacts are not included here. The source `metrics.json` SHA256 is `5fe9d8649f17392c5bacf7ddd7e6d89624c5ff543adb4c02ed62332bd9f765d7`; source `checkpoint.json` SHA256 is `daa7551693e0781b41f6c20fb1edf9f51a0a56c663ed4c33a8bf7c0e066ce4d0`. Text newline normalization can change file hashes without changing the recorded JSON values.

For the repeatable CUDA journey and its environment limits, use the [verification skill](../../.agents/skills/verify-async-rl-training/SKILL.md). CPU CI never downloads or loads a model.
