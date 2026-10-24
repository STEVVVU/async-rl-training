# Frozen heads on a fresh arithmetic holdout

Five previously selected clipped heads averaged **49.91% expected correct-answer probability**, compared with **40.20%** for the initial head, on 256 fresh operand pairs. The gain is **9.71 percentage points**. Individual clipped means range from **49.10% to 50.40%**. Every clipped seed improved; every validation-selected unclipped arm retained the initial head.

These are exact expectations under the constrained two-stage tool-and-digit policy. They are not observed generated-answer accuracy or a general LLM benchmark. All seeds share the same 256 questions.

## Protocol and result

The training and checkpoint selection from the [earlier repair](head-quality-20261024.md) remained frozen. No new optimization, checkpoint choice, threshold adjustment, or example filtering used this holdout.

The new protocol enumerates unordered pairs with operands from 10 through 99, shuffles them with seed 20261024, and takes the first 256. None overlaps the earlier single-digit train, validation, or test pairs. The task remains addition modulo ten with the existing sum and difference tools. This tests wider operands within that environment.

Before constructing prompts or forwarding the model, the evaluator checks the reviewed freeze digest, original selection and training receipts, every candidate checkpoint, source, cached features, and pretrained model files. It recomputes the original validation selections. The same initial head and all ten selections receive every pair in the same order.

| Seed | Clipped expected answer probability | Clipped expected reward |
| ---: | ---: | ---: |
| 11 | 0.502343 | 0.739894 |
| 23 | 0.504016 | 0.747697 |
| 37 | 0.498022 | 0.738331 |
| 51 | 0.499926 | 0.743082 |
| 71 | 0.491026 | 0.735497 |
| Mean | 0.499066 | 0.740900 |
| Initial head | 0.401979 | 0.590760 |

All 2,816 head/pair evaluations completed with finite results. That count is 256 shared questions evaluated by eleven heads, not 2,816 independent problems. The evaluator checks actual full-model logits on the first sixteen frozen pairs for each head. All 48 prompts per head matched both cached projections with zero recorded BF16 ULP error, within the unchanged one-ULP limit. The other 240 pairs use real frozen-backbone features without a separate full-model logit replay.

## Cost and reproduction

One GPU evaluation took **61.50 seconds** on an RTX 5080 Laptop GPU, Torch 2.11.0+cu128, with one Torch CPU thread. Peak Torch CUDA allocation was **1,527,737,856 bytes** and reserve was **1,556,086,784 bytes**. Peak parent working set was **2,164,191,232 bytes**. These are one-run resource observations, not a speedup comparison or total system memory.

The maintained [verification skill](../.agents/skills/verify-async-rl-training/SKILL.md#fresh-wider-operand-holdout) documents freeze and evaluate commands using `scripts/verify_fresh_head_holdout.py`. A new run requires the original native checkpoints and cached model assets. Weights are not distributed in Git.

The [freeze](evidence/fresh-head-holdout-20261024/freeze.json), [per-pair results](evidence/fresh-head-holdout-20261024/test.json), and [raw identities](evidence/fresh-head-holdout-20261024/raw-sha256.json) retain the complete experiment. Public text uses LF. The original v4 freeze SHA-256 is `8222d784045dfa3c7329d43aa84d80ccd4eb1df96fcb7b17056d34b221862a34`; the original result SHA-256 is `279ca0ecdebc933a4364a1220cf7cea82e36b72701091c65970a804ec7a5cda2`.

## Limits and retained failures

This is a synthetic, head-only arithmetic policy trained serially, with a frozen backbone and constrained actions. It does not establish async quality parity, distributed training throughput, full-model fine-tuning, or general language-model improvement. The original negative sync/async study and smaller reused holdout remain published.

Earlier freeze-only directories remain retained. The v3 invocation failed to import the repository modules before model loading. Its failure receipt remains intact; the import path was repaired and regression-tested before the separate v4 freeze and evaluation. No failed holdout examples were removed.
