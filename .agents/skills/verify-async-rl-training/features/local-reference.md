# Local pretrained async RL

## Sub-features

Cached pretrained Qwen weights, sampled tool action followed by an answer action, outcome reward, importance-corrected policy-gradient update, asynchronous queue, pause/update/resume, strict fence, and heldout reward curve.

## How to get to it (user POV)

Run the doctor and drive commands in the parent skill.

## Driving it with the CLI

Inspect events and metrics. Confirm optimizer steps, a nonzero LM-head parameter delta, mixed-version partial trajectories, a rejected straggler, bounded completion, and cleanup preserving evidence. Heldout reward need not increase.

## Gotchas

The constrained arithmetic action space and LM-head-only update prove mechanics, not general agent quality or distributed throughput. A cached pinned model and a CUDA GPU are prerequisites.
