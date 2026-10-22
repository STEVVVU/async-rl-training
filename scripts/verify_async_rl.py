"""Drive the local pretrained reference; never installs or downloads dependencies."""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots" / REVISION


def doctor():
    import torch
    import transformers

    ready = torch.cuda.is_available() and (MODEL / "model.safetensors").is_file()
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": str(MODEL),
        "revision": REVISION,
        "ready": ready,
        "native_skyrl": "unverified; requires supported Python >=3.12 Linux FSDP/vLLM/NCCL environment",
    }
    print(json.dumps(result, indent=2))
    if not ready:
        raise SystemExit("Cached safetensors and CUDA are required. No download attempted.")


def evidence(output):
    required = ["events.jsonl", "metrics.json", "policy-delta.safetensors", "checkpoint.json"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise SystemExit(f"Missing retained evidence: {missing}")
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    updates = [event for event in events if event["event"] == "update"]
    trained = [uid for update in updates for uid in update["trained"]]
    assert len(updates) == metrics["updates"] == checkpoint["version"] == 4
    assert len(trained) == len(set(trained)) == metrics["trained_groups"] == 8
    assert "straggler" not in trained and checkpoint["terminal_uids"]["straggler"] == "stale"
    assert metrics["parameter_delta_l2"] > 0 and metrics["actual_model_head_delta_l2"] > 0
    assert len(metrics["heldout_curve"]) == 5 and len(metrics["mixed_version_groups"]) == 3
    assert checkpoint["model_revision"] == REVISION
    for update in updates:
        index = events.index(update)
        assert events[index - 1]["event"] == "pause"
        assert events[index + 1]["event"] == "resume"
    restored = [event for event in events if event["event"] == "checkpoint_reload"]
    assert len(restored) == 1 and restored[0]["head_equal"] and restored[0]["skipped_terminal"] == 10
    print(json.dumps({"evidence": "passed", "retained": required, "updates": len(updates)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "drive", "evidence", "cleanup"])
    parser.add_argument("--output", type=Path, default=ROOT / ".evidence/local-reference")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.command == "doctor":
        doctor()
    elif args.command == "drive":
        doctor()
        from examples.local_async.train import run

        run(MODEL, args.output)
    else:
        evidence(args.output)
        if args.command == "cleanup":
            print("Short-lived drive has exited. No persistent process or scratch data to remove. Evidence retained.")


if __name__ == "__main__":
    main()
