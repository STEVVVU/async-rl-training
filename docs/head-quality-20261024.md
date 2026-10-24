# Resource-light head-quality repair

Global gradient clipping improved this bounded arithmetic policy after the earlier local study lost quality. Five validation-selected clipped heads achieved mean held-out expected reward **0.886297**, versus **0.758882** for the untouched head. Mean expected correct-answer probability increased from **56.29% to 64.33%**. Every clipped seed improved on the same ten reused held-out pairs. These are expectations under the constrained two-stage policy, not greedy accuracy, a general LLM benchmark, or fifty independent questions.

## Repair and protocol

An independent CPU oracle confirmed the existing PPO loss, signed clipping branches, gradient and row installation. A controlled cached-feature diagnostic instead reproduced large SGD head/logit excursions and rapid answer-probability collapse. The shared local update now uses native `torch.nn.utils.clip_grad_norm_` at global norm 0.5 before SGD, rejecting nonfinite norms while returning the preclip norm. The configuration records this setting. The comparison disables this operation alone; learning rate, reward, data, precision and update count stay fixed.

The norm limit is a predeclared optimizer setting consistent with [standard PPO practice](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), rather than a quality pass threshold. [PyTorch provides the operation](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html). It bounds the FP32 master-head SGD step; it does not guarantee a policy KL bound or convergence.

Training reused 135 hash-verified frozen backbone features for 35 train pairs and ten validation pairs, without loading the base model. Five seeds ran clipped and unclipped arms for 256 updates with two groups per update, one CPU thread, BF16 projections and an FP32 master head. Distinct group RNG identities and tool-before-answer draws follow the synchronous reference schedule. CPU RNG values are not claimed identical to CUDA.

All ten runs completed 256 updates and all 30 native checkpoints at steps 32, 128 and 256. Master/installed BF16 rows round-tripped. Selection used validation expected reward only, with the initial head eligible and earlier steps winning ties. All clipped arms selected step 32; all unclipped arms selected step zero. Source, configuration, inputs, candidates and selection were frozen before the separate model evaluation.

| Seed | Selected step, clipped / unclipped | Clipped test expected reward | Clipped expected correct-answer probability |
| --- | --- | ---: | ---: |
| 11 | 32 / 0 | 0.896638 | 0.657371 |
| 23 | 32 / 0 | 0.888610 | 0.643995 |
| 37 | 32 / 0 | 0.873214 | 0.631561 |
| 51 | 32 / 0 | 0.882093 | 0.637851 |
| 71 | 32 / 0 | 0.890928 | 0.645625 |

The GPU phase verified cached Qwen3-0.6B assets, tokenizer IDs and original digit rows, detached the output head from tied embeddings, then constructed test prompts. Input/source/candidate checks and selection recomputation preceded that access. Initial and all ten candidate logit checks equaled full-model logits in both projections, within the unchanged one-BF16-ULP limit. No test result changed the training or selection settings.

The raw [training](evidence/head-quality-20261024/training.json), [selection](evidence/head-quality-20261024/selection.json), and [test](evidence/head-quality-20261024/test.json) receipts retain every outcome and per-pair result. Native weights stay outside Git; their [hashes](evidence/head-quality-20261024/checkpoint-hashes.json) remain. Public JSON uses LF; [original identities](evidence/head-quality-20261024/raw-sha256.json) record reversible line endings, and [published hashes](evidence/head-quality-20261024/sha256.json) cover the copies.

## Resources and reproduction

The CPU sweep took **7.27 seconds before receipt writing**, including cache loading, updates, validation and checkpoint saves. Parent peak working set, including imports, was **640,978,944 bytes**. Interpreter startup is outside that timer. The separate GPU evaluation took **16.77 seconds**, including checks, model loading, feature preparation and evaluation. Its parent peak was **2,163,113,984 bytes**. Torch CUDA peak allocation/reservation was **1,535,064,064 / 1,564,475,392 bytes**, excluding non-Torch driver memory. These are resource observations, not an async speedup comparison.

Runtime was Python 3.11.9, Torch 2.11.0+cu128, Transformers 5.20.2, and RTX 5080 Laptop GPU for evaluation. The [verification skill](../.agents/skills/verify-async-rl-training/SKILL.md#resource-light-head-quality-comparison) gives the explicit train/test commands. Use a new output directory and review selection before testing. Successful test receipts cannot be overwritten. Full-study memory guards remain unchanged; this path reuses the supported model-loading API without a new package.

Default GPU Doctor/Drive/Evidence/Cleanup also passed after clipping, preserving staleness fencing, nonzero weight updates, exact checkpoint reload and evidence retention. Dependency-free and separate Torch analytic/contract tests passed. CI runs dependency-free behavior and static checks, without claiming CUDA or Torch-only test execution.

## Limits and retained failures

This is head-only, serial CPU training followed by GPU evaluation on a small previously used arithmetic holdout. It does not establish native distributed SkyRL performance, async quality parity, full-model fine-tuning, generated-answer accuracy, statistical significance or broad generalization. The original negative [CUDA sync/async study](matched-study-results-20261022.md) remains unchanged.

Earlier prototypes and partial runs remain evidence. One early probe's setup log was overwritten; its replacement is explicitly labeled as a reconstruction rather than an original receipt. Main corrected a prototype's duplicated RNG identity/draw order before this study. The first GPU launch stopped before test-prompt construction because optional direct placement required unavailable Accelerate; the final driver reuses the existing model-loading API. Formatting and added inventory/runtime guards required fresh training receipts, without test-driven retuning. Partial evidence cleanup was blocked and was not retried.
