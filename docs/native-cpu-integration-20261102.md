# Native CPU fence and checkpoint integration

[Hosted run 41415003904](https://github.com/STEVVVU/async-rl-training/actions/runs/41415003904) passed all 15 tests in `tests/train/test_fully_async_trainer.py` in 13.25 seconds at commit `58d8f1874391d26c637b991c5003dd85850059bf`. The retained workflow artifact includes the test log, generated dependency manifest, and checkpoint files. The test-only Ubuntu environment used PyTorch 2.13.0+cpu, Ray 2.57.0, and torchdata 0.11.0+cpu. It imported `FullyAsyncRayPPOTrainer` from the project.

The integration case constructs trainer instances with `__new__` and supplies their required state. It passes a step-zero group, a duplicate of that group, and two eligible groups through the trainer's actual queue drain and strict fence at step two. The stale UID `0` is filtered once, the kept UIDs are `1` and `2`, and available capacity rises from three to four. The test manually marks the kept UIDs consumed.

The trainer's base and fully async checkpoint methods write `data.pt`, `trainer_state.pt`, and `fully_async_state.pt`. The retained checkpoint records global step 1 and epoch 0. Its consumed UID set is `{0, 1, 2}` and its filtered set is `{0}`. A fresh trainer runs the real checkpoint load and the resume prologue, restores the trained count to two, and draws `3`, `4`, and `5` next. The test stops at weight synchronization.

Model checkpoint dispatch is mocked, so the test writes no policy weights or optimizer state. It does not perform an optimizer update, GPU inference, a full trainer initialization, or distributed training and inference overlap. The local head-quality studies exercise a separate cached-feature model path and do not prove these native behaviors.

[Earlier run 41414591827](https://github.com/STEVVVU/async-rl-training/actions/runs/41414591827) installed the CPU environment and imported the trainer. Fourteen tests passed, while the integration case failed before queue processing because it initialized Ray after the session fixture had already done so. The passing run uses the fixture's Ray instance. Both runs set `RAY_ADDRESS=local`.

The job builds its test environment from locked versions and hashes, omits GPU distributions, and substitutes the published CPU build of the same PyTorch version. It does not validate the full production dependency set or GPU backends. See the [workflow](../.github/workflows/local_cpu.yaml) and [test](../tests/train/test_fully_async_trainer.py) for the executable setup.
