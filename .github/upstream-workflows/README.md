# Preserved upstream workflow reference

These sixteen YAML files preserve SkyRL's workflow contents at commit `b3046522ffb63738eed75a9f4853b92cc161b0ff`. Moving added a final newline to `cpu_skyrl.yaml` and `gpu_e2e_ci_megatron.yaml`; the other fourteen Git blobs are exact copies. GitHub does not execute workflows in this directory. They remain available for a supported upstream deployment, including its original CPU suites and Anyscale GPU jobs.

The personal checkout uses [local CPU CI](../workflows/local_cpu.yaml). It runs dependency-free behavioral checks, lint, formatting, and static compilation. It uploads previously captured pretrained-model metrics with an explicit statement that the CI job does not run the model or native distributed backend. Native test commands and all upstream tests remain intact.

Moving the inherited workflows prevents push, pull-request-target, label, or manual events from invoking upstream-specific integrations in this checkout. Restoring them requires a deliberate environment and workflow review; no credentials or external compute are configured by local CI.
