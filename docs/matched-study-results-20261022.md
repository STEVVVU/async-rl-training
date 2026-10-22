# Matched local RL study, 22 October 2026

The study completed at source `059ae60ef128346701cb7f260da183f1c562e970`. The result is negative for quality. Every validation-selected policy scored below the untouched initial model on the ten held-out arithmetic pairs. There is no demonstrated async quality or speed advantage.

| Metric | Initial head | Sync, five seeds | Async, five seeds |
| --- | ---: | ---: | ---: |
| Mean held-out expected reward | 0.758882 | 0.331993 | 0.331289 |
| Mean held-out answer-success probability | 56.2921% | 8.2024% | 8.1534% |
| Per-seed expected-reward range | Same initial head | 0.254001-0.485759 | 0.254001-0.430525 |
| Accepted groups per run | No training | 512 | 512 |
| Generated groups per run | No training | 512 | 514 |
| Sampled action tokens per run | No training | 2,048 | 2,056 |
| Training-only time, median (range) | No training | 2.847 s (2.475-3.286) | 2.811 s (2.513-3.152) |

The timing ranges overlap. Training-only timing excludes model load, feature preparation, validation, full-model consistency checks, and checkpoint writes. All ten training runs together took 203.699 seconds in the external supervisor, including those checks. This is a head-only experiment with one event loop and serialized CUDA kernels. It does not measure native distributed rollout, GPU/rollout overlap, production throughput, or an end-to-end speedup. No profiler isolates the small timing difference, so it is inconclusive.

## Frozen method

The model is cached Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, weight SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`. Only ten output-head rows, 10,240 parameters, train. The backbone and input embeddings stay frozen. The task samples one of two owned arithmetic tools and a decimal answer. Reported answer success is the model-distribution probability of the correct answer, not greedy accuracy or observed sampled test success. Expected reward also includes a 0.25 bonus for selecting the correct tool.

The 55 canonical unordered digit pairs were shuffled once with seed 421 and split 35/10/10 into train, validation, and test. Seeds 11, 23, 37, 51, and 71 use paired schedules, with arm order reversed on alternating seeds. Each run accepts 512 groups in 256 two-group updates. Async carries actions across one version, admits 510 mixed-version groups, and retains two generated tail groups. The extra eight action tokens per async run remain counted; generated work is not identical between arms. No stale groups appeared in this schedule; separate default-reference checks exercise stale rejection.

Both arms use installed BF16 model rows and cached BF16 features for sampling/evaluation, float32 sampling logits, and an FP32 master head with differentiable BF16 update forwards. Learning rate 0.005, reward baseline 0.5, clipping 0.8-1.2, update count and maximum policy age one were unchanged. Checkpoints at steps 32, 128, and 256 were selected by validation expected reward only, with the earliest step breaking ties. The initial head was the comparison baseline, not a selectable trained checkpoint. All 30 candidate checkpoint hashes were frozen before the separate final-test action read test prompts. No tuning followed the held-out result.

## Verification and retained evidence

Preparation matched full-model BF16 digit logits exactly for both two-row and ten-row projections over all 135 train/validation prompts. The one-ULP gate remained active for initial, checkpoint and test checks. All ten runs passed full accepted/tail/action/version ledgers. The main audit independently recomputed run/metric/data/source hashes, selected steps, and native safetensors file/tensor hashes, and reconciled all 100 final per-pair score records with their aggregates. It did not rerun final inference. All 30 checkpoints loaded exactly, with installed BF16 rows equal to the rounded FP32 master and identical initial rows across runs. Actual parameter deltas are retained per checkpoint.

The original four-update real-model Drive, Evidence and Cleanup passed again at `.evidence/reference-regression-20261022-1051`. The 20 dependency-free tests, pinned Ruff/Black checks and [CPU CI at the measured source](https://github.com/STEVVVU/async-rl-training/actions/runs/39765369213) passed. CPU CI does not run the GPU study. A lint finding about loop-carried state was resolved with explicit initialization and an invariant assertion before freezing this source; no learning or quality threshold changed.

The primary study directory is `.evidence/matched-study-final-20261022-104451`. It retains preparation, manifest, prompt/features, run order, ten run receipts, all metrics/checkpoints, frozen selection, final-test receipt and adjacent supervisor logs. The [reviewed measurement record](evidence/matched-study-20261022/measurement.json) preserves source/model identities, primary hashes, per-seed values and checkpoint audits. The earlier `.evidence/matched-study-20261022-103857` compatibility run is preserved but excluded from these results. Earlier missing-Torch and insufficient-RAM attempts also remain preserved.

The environment was Windows build 26200, Python 3.11.9, Torch 2.11.0+cu128, Transformers 5.20.2 and an RTX 5080 Laptop GPU. Resource guards required 8 GiB available host memory, 4 GiB free CUDA memory and 10 GiB free disk at launch; these guards are not measured peak-usage figures. Every owned stage exited, released CUDA and left its evidence intact. No user process, model cache or shared runtime was modified.

Ten test pairs shared across five seeds are ten unique held-out questions. These results do not establish broader language quality, benchmark accuracy, production reliability, native Linux SkyRL operation, a 24-worker gain, or the resume's worker-idle-time targets. A new quality study would need a prospectively frozen method and fresh held-out data.
