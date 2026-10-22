# Strict production fence

## Sub-features

Opt-in age bound, group-atomic stale discard, UID duplicate suppression, filtered accounting, finite epoch exhaustion, completed-step checkpoint numbering, zero-update UID restoration, and checkpoint resume capacity alignment.

## How to get to it (user POV)

Set `trainer.fully_async.strict_staleness=true` in the normal fully async upstream entrypoint.

## Driving it with tests

Run `uv run --isolated --extra skyrl-train --extra dev pytest tests/train/test_fully_async_trainer.py` on a supported Linux environment. On Windows set `PYTHONPATH=.` and run the dependency-free suite with `uv run --isolated --no-project --python <interpreter> -m unittest discover -s tests/local_async -v`. Model stages require the explicit-interpreter command in the verification skill.

## Gotchas

Stale groups are permanently filtered for the current epoch, never retried. Too many stale groups can shorten the epoch. The scheduling stamp is a conservative oldest-policy bound across partial continuations. Dependency-free native-body checks compile selected source methods and do not prove full trainer import, dispatch, or distributed execution. FSDP/vLLM/NCCL checks require Linux and suitable hardware. The native test is added but unexecuted here.
