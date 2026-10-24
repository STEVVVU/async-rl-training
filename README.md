# Async RL training with strict policy-age fencing

This project adds opt-in strict policy-age fencing and a bounded pretrained-model reference run to the complete Apache-2.0 [SkyRL](https://github.com/NovaSky-AI/SkyRL) framework. It preserves the full upstream source and history at commit `b3046522ffb63738eed75a9f4853b92cc161b0ff`, all licenses and notices, distributed trainers, generators, and examples. The original [upstream README](README.upstream.md) is preserved verbatim.

The coursework source is [Asynchronous Agentic Reinforcement Learning](https://people.eecs.berkeley.edu/~kubitron/courses/cs262a-F25/projects/reports/project1008_paper_ver2_72488057446561351955.pdf), by Charlie Ruan, Hanchen Li, and Jiaxiang Yu for Berkeley CS262a Fall 2025. The [official fully async tutorial](https://docs.skyrl.ai/docs/tutorials/fully_async) describes the framework mechanism. This checkout is based on current official SkyRL, not a verified copy of the students' exact experimental branch. Their scaling-law and throughput experiments have not been reproduced.

## Resource-light quality repair

The [24 October head-quality comparison](docs/head-quality-20261024.md) fixes oversized local head updates with native gradient-norm clipping. Across five seeds, validation-selected policies increased mean expected correct-answer probability from **56.29% to 64.33%** and expected reward from **0.758882 to 0.886297** on ten reused held-out arithmetic pairs. These are constrained-policy expectations, not general LLM accuracy or fifty independent questions. The original negative sync/async study remains below.

The complete ten-run CPU sweep used one thread, **7.27 seconds** before receipt writing and **611 MiB** peak parent memory. It reused frozen pretrained features without loading the model. A separate model/logit evaluation took 16.77 seconds and about 1.43 GiB of Torch CUDA allocation. The report retains all seeds, 30 checkpoint hashes, model-logit checks, resources and reproducible commands. No distributed throughput or speedup is claimed.

## Enable strict fencing in SkyRL

Use the existing fully async entrypoint and normal model, data, and cluster configuration. Add `trainer.fully_async.strict_staleness=true`. For example, adapt the retained `examples/train/fully_async/fully_async_run_gsm8k.sh` and keep its backend settings.

`max_staleness_steps` normally limits aggregate producer capacity. A long-running group can exceed that age. Strict mode checks each completed group's scheduling step against the consumer step before admitting it to a mini-batch. The scheduling stamp conservatively bounds its oldest token policy, including continuations after in-flight updates. Groups exactly at the age limit remain eligible.

A stale group is terminal for its UID for the current epoch. Its capacity is returned and the existing filtered-UID checkpoint ledger prevents regeneration on resume. A duplicate queue delivery does not repeat those effects. An exhausted epoch filters its partial final mini-batch and can train fewer steps. Exhaustion and final checkpoints record the last completed update, including zero, while ordinary post-update checkpoints retain their normal step number. Zero-update checkpoint reload also restores filtered UIDs. The default remains `strict_staleness=false`, preserving warning-only admission.

## Run the local pretrained reference

This separate backend runs on the observed Windows Python 3.11 environment. It uses real cached [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md) safetensors at revision `c1899de289a04d12100db370d81485cdf75e47ca`, licensed Apache-2.0. It requires CUDA and the versions recorded in [requirements.txt](examples/local_async/requirements.txt). It never downloads a model, enables remote Python, or installs into the shared ML environment.

```powershell
$uv = 'C:/Users/steve/.codex/tools/uv-0.13.4/uv-0.13.4.data/scripts/uv.exe'
$ml = 'C:/Users/steve/.codex/tools/coursework-ml/Scripts/python.exe'
& $uv run --isolated --no-project --python $ml -- cmd.exe /d /c $ml scripts/verify_async_rl.py doctor
& $uv run --isolated --no-project --python $ml -- cmd.exe /d /c $ml scripts/verify_async_rl.py drive --output .evidence/my-new-run
& $uv run --isolated --no-project --python $ml -- cmd.exe /d /c $ml scripts/verify_async_rl.py evidence --output .evidence/my-new-run
& $uv run --isolated --no-project --python $ml -- cmd.exe /d /c $ml scripts/verify_async_rl.py cleanup --output .evidence/my-new-run
```

Choose an unused output directory. Coordinate exclusive CUDA access before driving. The Windows `cmd.exe` invocation explicitly launches the existing ML interpreter without resolving SkyRL or modifying its environment. Passing the Python script directly to `uv run --isolated --python $ml` creates an environment without those ML packages. The documented command does not isolate the existing ML packages. The project itself requires Python >=3.12 and different dependency pins.

The reference freezes the pretrained Transformer and trains ten existing output-head rows, with 10,240 trainable parameters. It unties the head so input embeddings stay frozen. Actions choose one of two owned arithmetic tools and then a decimal answer. Each group has two sampled trajectories. Async tasks retain partial trajectories across weight updates, and an asyncio queue delivers completed groups to the shared strict fence. Paused tasks interleave cooperatively on one GPU; this is not a measurement of concurrent kernel throughput.

The loss uses sampled outcome rewards, a fixed baseline, stored action log probabilities, and a clipped importance ratio. No supervised target loss drives the update. Frozen context features can be cached because only the output head changes. Float32 head rows drive sampling and optimization; their updates are also copied into the real model's bfloat16 output head. The checkpoint saves exact float32 rows and their delta in safetensors, plus versions and terminal UIDs in JSON. Reload verifies exact head equality, skips terminal UIDs, and generates a fresh trajectory. Stateless SGD needs no optimizer tensor checkpoint.

The retained October 21, 2026 run in `.evidence/final-reference/` completed four updates and trained eight groups. Three trained groups contain action versions `[0,1]`, `[1,2]`, and `[2,3]`. The straggler with versions `[0,2]` was filtered once. The real model head changed by L2 `0.0690855`. Heldout expected reward over four disjoint arithmetic prompts changed from `0.760472` to `0.543493`, with the full five-point curve in `metrics.json`. This small run demonstrates mechanics and showed a reward decrease. It makes no convergence, quality, speedup, or scaling-law claim.

The [completed five-seed study](docs/matched-study-results-20261022.md) uses a frozen 55-pair manifest with 35/10/10 train/validation/test splits and 256 updates per run. All ten sync/async runs passed accounting, precision, and native checkpoint reload checks; all 30 checkpoints were independently rehashed and loaded. Validation selected among steps 32, 128, and 256 before the separate held-out test. Every selected policy underperformed the untouched model on the ten test pairs. Initial expected reward was 0.758882, versus mean 0.331993 for sync and 0.331289 for async across five seeds. Mean answer-success probability fell from 56.29% to 8.20% and 8.15%. This is a negative quality result, with no demonstrated async quality or speed advantage.

This study trains only ten output-head rows with BF16 cached features and installed BF16 model rows, an FP32 master head, and float32 sampling logits. Sync and async accept 512 groups per run; async generates two additional tail groups and retains their cost. One asyncio event loop serializes CUDA kernels. Preparation matched full-model digit logits exactly for all 135 train/validation prompts, and later checks retained the one-BF16-ULP gate. Training-only timing excludes model loading, validation, consistency checks and checkpoint writes. Neither this local study nor the earlier reference proves native distributed SkyRL throughput or a 24-worker gain.

## Verify changes

The maintained [verification skill](.agents/skills/verify-async-rl-training/SKILL.md) and [feature map](.agents/skills/verify-async-rl-training/features/README.md) describe the complete local journey and the native gap.

```powershell
$env:PYTHONPATH='.'
& $uv run --isolated --no-project --python $ml -m unittest discover -s tests/local_async -v
```

The twenty dependency-free checks cover fence boundaries, duplicate delivery, finite exhaustion, default warning-only behavior, rejection capacity, trained-versus-filtered resume accounting, checkpoint step/capacity alignment including zero updates and partial exhaustion, reference-worker startup failure, and matched-study split, schedule, RNG, action-token accounting, heldout prompt isolation, run-inventory, and selection-receipt invariants. Some compile selected source method bodies with controlled IO without importing their dependencies. Those are source-sliced unit tests, not full trainer integration. The added native test in `tests/train/test_fully_async_trainer.py` uses the real imported trainer and dataloader on a supported environment.

[Local CPU CI](.github/workflows/local_cpu.yaml) runs those tests and static/lint checks without loading a model. Its downloadable artifact contains [retained real CUDA measurements](verification/pretrained-reference/README.md), including the reward decline, and is explicitly labeled historical evidence. Original upstream workflow contents remain in [.github/upstream-workflows](.github/upstream-workflows/README.md), where their Anyscale and upstream-specific triggers do not execute.

Native validation remains unexecuted. This Windows machine's shared ML interpreter has no Ray or torchdata and uses Python 3.11 and Transformers 5.20.2, outside upstream's pins. An offline attempt to run the upstream CPU command failed before collection on uncached `polars-runtime-32`. Full training additionally needs the supported Linux FSDP/vLLM/NCCL setup. No distributed or production checkpoint execution is claimed.

The isolated uv 0.13.4 tool came from official PyPI. Its retained wheel SHA256 is `b2ff80397273fff0e02c20eacae6882834c811827d6fc08f658db156fe754386`. Its Apache/MIT license metadata remains alongside the tool. No project dependency was added. The shared Torch environment has the documented unused `torch.jit.script` advisory `GHSA-rrmf-rvhw-rf47`; this path does not invoke TorchScript, and no zero-vulnerability claim is made.
