"""Drive the local pretrained reference; never installs or downloads dependencies."""

import argparse
import ctypes
import json
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots" / REVISION


def capacity_guard(path):
    free_disk = shutil.disk_usage(path).free
    if free_disk < 10 * 1024**3:
        raise SystemExit(f"Study requires 10 GiB free disk; found {free_disk / 1024**3:.1f} GiB")
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise SystemExit("Could not read available system memory")
        available = status.avail_phys
    else:
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    if available < 8 * 1024**3:
        raise SystemExit(f"Study requires 8 GiB available system memory; found {available / 1024**3:.1f} GiB")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the matched study")
    free_vram, _ = torch.cuda.mem_get_info(0)
    if free_vram < 4 * 1024**3:
        raise SystemExit(f"Study requires 4 GiB free CUDA memory; found {free_vram / 1024**3:.1f} GiB")


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
    parser.add_argument(
        "command",
        choices=["doctor", "drive", "evidence", "cleanup", "study-prepare", "study-run", "study-select", "study-test"],
    )
    parser.add_argument("--output", type=Path, default=ROOT / ".evidence/local-reference")
    parser.add_argument("--study", type=Path, default=ROOT / ".evidence/matched-study")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.command == "doctor":
        doctor()
    elif args.command == "drive":
        doctor()
        from examples.local_async.train import run

        run(MODEL, args.output)
    elif args.command == "study-prepare":
        doctor()
        capacity_guard(args.study.parent)
        from examples.local_async.train import prepare_study

        prepare_study(MODEL, args.study)
    elif args.command == "study-run":
        doctor()
        capacity_guard(args.study)
        from examples.local_async.train import run_study

        run_study(MODEL, args.study)
    elif args.command == "study-select":
        from examples.local_async.train import select_study

        select_study(MODEL, args.study)
    elif args.command == "study-test":
        doctor()
        capacity_guard(args.study)
        from examples.local_async.train import test_study

        test_study(MODEL, args.study)
    else:
        evidence(args.output)
        if args.command == "cleanup":
            print("Short-lived drive has exited. No persistent process or scratch data to remove. Evidence retained.")


if __name__ == "__main__":
    main()
