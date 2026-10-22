# Feature map

| Feature | Driver | Acceptance |
| --- | --- | --- |
| [Local pretrained async RL](local-reference.md) | `scripts/verify_async_rl.py` | Tool trajectories, pause/update/resume, policy delta and retained metrics |
| [Strict production fence](strict-fence.md) | Focused tests and upstream trainer | Oldest group version is checked before training; Linux distributed run remains required |
| Framework-first metrics preflight | Existing core verification | The core/fence/checkpoint preflight is retained; it does not prove a multi-seed study |
| [Matched head-only study, draft](metrics.md) | `scripts/verify_async_rl.py study-*` | Frozen 35/10/10 manifest, five paired seeds, validation-only selection, hash-gated separate test action; unexecuted |
