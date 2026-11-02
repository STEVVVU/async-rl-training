# Strict production fence

## Sub-features

Opt-in age bound, group-atomic stale discard, UID duplicate suppression, filtered accounting, finite epoch exhaustion, completed-step checkpoint numbering, zero-update UID restoration, and checkpoint resume capacity alignment.

## How to get to it (user POV)

Set `trainer.fully_async.strict_staleness=true` in the normal fully async upstream entrypoint.

## Driving it with tests

Run `uv run --isolated --extra skyrl-train --extra dev pytest tests/train/test_fully_async_trainer.py` on a supported Linux environment. On Windows set `PYTHONPATH=.` and run the dependency-free suite with `uv run --isolated --no-project --python <interpreter> -m unittest discover -s tests/local_async -v`. Model stages require the explicit-interpreter command in the verification skill.

## Gotchas

Stale groups are permanently filtered for the current epoch, never retried. Too many stale groups can shorten the epoch. The scheduling stamp is a conservative oldest-policy bound across partial continuations. Dependency-free native-body checks compile selected source methods and do not prove full trainer import, dispatch, or distributed execution. The `native_fence_integration` Ubuntu CI job exports versions and hashes from `uv.lock`, removes GPU-only distributions, and substitutes the official CPU build of the same PyTorch version with its published hash. The resulting test-only environment is not the production dependency set. Its first bounded install and import have not yet run. The job is configured to import the actual trainer, drain a real queue, write and reload actual checkpoint files, and run the resume prologue with model dispatch and weight synchronization substituted. It stops before training and does not test an optimizer update, GPU inference, distributed workers, or overlap. FSDP/vLLM/NCCL checks require Linux and suitable hardware.
