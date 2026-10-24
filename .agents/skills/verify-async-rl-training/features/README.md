# Feature map

| Feature | Driver | Acceptance |
| --- | --- | --- |
| [Local pretrained async RL](local-reference.md) | `scripts/verify_async_rl.py` | Tool trajectories, pause/update/resume, policy delta and retained metrics |
| [Strict production fence](strict-fence.md) | Focused tests and upstream trainer | Oldest group version is checked before training; Linux distributed run remains required |
| Framework-first metrics preflight | Existing core verification | The core/fence/checkpoint preflight is retained; it does not prove a multi-seed study |
| [Matched head-only study](metrics.md) | `scripts/verify_async_rl.py study-*` | Ten runs, 30 native checkpoints, validation-only selection and separate held-out test passed mechanism checks; all selected policies lost quality versus the initial head. [Results](../../../../docs/matched-study-results-20261022.md) |
| [Resource-light head-quality comparison](../../../../docs/head-quality-20261024.md) | `scripts/verify_head_quality.py train/test` | Ten CPU runs and 30 checkpoint hashes passed; separate GPU evaluation verified model logits. Mean expected answer probability 56.29% to 64.33% over ten reused held-out pairs, with no distributed or general-accuracy claim. |
