"""Dependency-free invariants for the opt-in matched-study adapter."""

import ast
import hashlib
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "examples/local_async/train.py"


def source_functions(*names):
    tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
    wanted = set(names)
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id
            in {
                "STUDY_SEEDS",
                "STUDY_CHECKPOINTS",
                "STUDY_UPDATES",
                "STUDY_GROUPS_PER_UPDATE",
                "STUDY_ACTIONS_PER_GROUP",
            }
            for target in node.targets
        )
    ]
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"hashlib": hashlib, "random": random, "json": json, "math": math, "Path": Path}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=constants + funcs, type_ignores=[])), str(TRAIN), "exec"),
        namespace,
    )
    return namespace


def synthetic_metrics(arm, seed=11, order_index=0):
    updates = 256
    accepted = {f"{i:04d}" for i in range(512)}
    tails = {"0512", "0513"} if arm == "async" else set()
    uids = accepted | tails
    versions, records = {}, {}
    for uid in accepted:
        group = int(uid)
        version = group // 2
        current = [version] * 4 if arm == "sync" or version == 0 else [version - 1] * 2 + [version] * 2
        versions[uid] = current
        records[uid] = {"decision": "keep", "oldest": min(current), "consumer_version": version, "versions": current}
    for uid in tails:
        versions[uid] = [255, 255, 256, 256]
        records[uid] = {"decision": "tail", "oldest": 255, "consumer_version": 256, "versions": versions[uid]}
    prompt_counts = {uid: 12 for uid in uids}
    action_counts = {uid: 4 for uid in uids}
    return {
        "run_id": f"{arm}-seed-{seed}",
        "run_order_index": order_index,
        "arm": arm,
        "seed": seed,
        "updates": updates,
        "counts": {
            "generated": len(uids),
            "accepted": 512,
            "stale": 0,
            "tail": len(tails),
            "action_tokens": len(uids) * 4,
            "prompt_tokens": sum(prompt_counts.values()),
            "mixed": 510 if arm == "async" else 0,
        },
        "group_versions": versions,
        "terminal_records": records,
        "prompt_tokens_by_uid": prompt_counts,
        "action_tokens_by_uid": action_counts,
        "losses": [0.1] * updates,
        "initial_validation": {
            "expected_reward": 0.4,
            "answer_success": 0.3,
            "per_pair": [{"expected_reward": 0.4, "answer_success": 0.3}],
        },
        "validation": [
            {
                "step": step,
                "expected_reward": 0.5,
                "answer_success": 0.4,
                "per_pair": [{"expected_reward": 0.5, "answer_success": 0.4}],
            }
            for step in (32, 128, 256)
        ],
        "timings": {"training_seconds": 1.0},
        "consistency_checks": {
            "initial": {"max_ulp": 1.0},
            "32": {"max_ulp": 1.0},
            "128": {"max_ulp": 1.0},
            "256": {"max_ulp": 1.0},
        },
        "checkpoint_receipts": {
            str(step): {
                "filename": f"step-{step}.safetensors",
                "sha256": "x",
                "head_sha256": "x",
                "installed_bf16_sha256": "x",
                "initial_head_sha256": "x",
            }
            for step in (32, 128, 256)
        },
        "config": None,
        "run_passed": True,
        "manifest_sha256": "manifest",
        "source_sha256": "source",
        "model_sha256": "model",
        "data_sha256": {"manifest": "manifest", "features": "features", "prompts": "prompts", "preparation": "prep"},
        "schedule_sha256": hashlib.sha256(json.dumps([[0, 0]], separators=(",", ":")).encode()).hexdigest(),
        "runtime": {"python": "test"},
        "model_config": {"model_type": "test"},
        "trainable_parameters": 10,
        "training_policy": "test",
        "runtime_policy": "test",
    }


class MatchedStudyTests(unittest.TestCase):
    def test_manifest_partitions_canonical_pairs_without_leakage(self):
        units = source_functions("study_manifest")
        manifest = units["study_manifest"]()
        partitions = manifest["partitions"]
        self.assertEqual([len(partitions[key]) for key in ("train", "validation", "test")], [35, 10, 10])
        flattened = [tuple(pair) for name in ("train", "validation", "test") for pair in partitions[name]]
        self.assertEqual(len(set(flattened)), 55)
        self.assertEqual(set(flattened), {(a, b) for a in range(10) for b in range(a, 10)})
        self.assertTrue(all(a <= b for a, b in flattened))
        self.assertEqual(manifest, units["study_manifest"]())

    def test_seed_schedule_and_group_rng_identity_are_stable(self):
        units = source_functions("study_manifest", "study_schedule", "study_run_order", "group_seed")
        left = units["study_schedule"](23)
        self.assertEqual(len(left), 512)
        self.assertEqual(left, units["study_schedule"](23))
        self.assertEqual(
            {tuple(pair) for pair in left[:35]},
            {tuple(pair) for pair in units["study_manifest"]()["partitions"]["train"]},
        )
        self.assertNotEqual(units["group_seed"](23, "0007"), units["group_seed"](23, "0008"))
        order = units["study_run_order"]()
        self.assertEqual(len(order), 10)
        self.assertEqual(
            [(item["seed"], item["arm"]) for item in order[:4]],
            [(11, "sync"), (11, "async"), (23, "async"), (23, "sync")],
        )

    def test_action_budget_counts_generated_partial_and_answer_actions_once(self):
        units = source_functions("record_action_tokens")
        for arm, generated, expected in (("sync", 512, 2048), ("async", 514, 2056)):
            totals, per_uid = {"action_tokens": 0}, {}
            for index in range(generated):
                uid = f"{index:04d}"
                units["record_action_tokens"](totals, per_uid, uid, 2)
                units["record_action_tokens"](totals, per_uid, uid, 1)
                units["record_action_tokens"](totals, per_uid, uid, 1)
            self.assertEqual(totals["action_tokens"], expected, arm)
            self.assertEqual(sum(per_uid.values()), expected, arm)
            self.assertTrue(all(value == 4 for value in per_uid.values()))

    def test_test_prompts_are_outside_prepared_features_and_map_after_extension(self):
        units = source_functions(
            "study_manifest", "_study_prompts", "_study_partition_prompts", "_prompts", "tool", "_prompt_keys"
        )
        manifest = units["study_manifest"]()
        frozen = units["_study_partition_prompts"](manifest, "train", "validation")
        test = units["_study_partition_prompts"](manifest, "test")
        self.assertTrue(set(frozen).isdisjoint(test))
        prompts = {f"p{i:03d}": prompt for i, prompt in enumerate(frozen)}
        for index, prompt in enumerate(test, start=len(prompts)):
            prompts[f"p{index:03d}"] = prompt
        by_text = units["_prompt_keys"](prompts)
        self.assertEqual(set(by_text), set(frozen) | set(test))
        self.assertTrue(all(by_text[prompt] in prompts for prompt in test))

    def test_selection_requires_all_run_validation_and_checkpoint_receipts(self):
        units = source_functions(
            "study_config", "study_manifest", "study_schedule", "validate_selection_run", "validate_study_metrics"
        )
        metrics = synthetic_metrics("async")
        metrics["schedule_sha256"] = hashlib.sha256(
            json.dumps(units["study_schedule"](11), separators=(",", ":")).encode()
        ).hexdigest()
        metrics["config"] = units["study_config"]()
        receipt = {
            "run_id": metrics["run_id"],
            "run_order_index": 0,
            "run_passed": True,
            "manifest_sha256": "manifest",
            "source_sha256": "source",
            "model_sha256": "model",
            "config": metrics["config"],
            "checkpoint_receipts": metrics["checkpoint_receipts"],
            "arm": metrics["arm"],
            "seed": metrics["seed"],
            "updates": metrics["updates"],
            "counts": metrics["counts"],
            "data_sha256": metrics["data_sha256"],
            "schedule_sha256": metrics["schedule_sha256"],
            "runtime": metrics["runtime"],
            "model_config": metrics["model_config"],
            "trainable_parameters": metrics["trainable_parameters"],
            "training_policy": metrics["training_policy"],
            "runtime_policy": metrics["runtime_policy"],
        }
        entry = {"arm": "async", "seed": 11, "index": 0}
        self.assertTrue(
            units["validate_selection_run"](
                metrics, receipt, entry, "manifest", "source", "model", metrics["data_sha256"]
            )
        )
        metrics["data_sha256"] = {**metrics["data_sha256"], "features": "old-features"}
        receipt["data_sha256"] = metrics["data_sha256"]
        with self.assertRaises(ValueError):
            units["validate_selection_run"](
                metrics,
                receipt,
                entry,
                "manifest",
                "source",
                "model",
                {
                    "manifest": "manifest",
                    "features": "features",
                    "prompts": "prompts",
                    "preparation": "prep",
                },
            )
        metrics["data_sha256"] = {
            "manifest": "manifest",
            "features": "features",
            "prompts": "prompts",
            "preparation": "prep",
        }
        receipt["data_sha256"] = metrics["data_sha256"]
        del metrics["validation"][-1]
        with self.assertRaises(ValueError):
            units["validate_selection_run"](
                metrics,
                receipt,
                entry,
                "manifest",
                "source",
                "model",
                {
                    "manifest": "manifest",
                    "features": "features",
                    "prompts": "prompts",
                    "preparation": "prep",
                },
            )

    def test_selection_inventory_requires_all_ten_interleaved_runs(self):
        units = source_functions("study_run_order", "_validate_run_inventory")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            order = units["study_run_order"]()
            for entry in order:
                (root / f"{entry['arm']}-seed-{entry['seed']}").mkdir()
            self.assertTrue(units["_validate_run_inventory"](root, order))
            (root / "sync-seed-71").rmdir()
            with self.assertRaises(ValueError):
                units["_validate_run_inventory"](root, order)

    def test_selection_run_rejects_wrong_action_totals_and_missing_checkpoint_identity(self):
        units = source_functions(
            "study_config", "study_manifest", "study_schedule", "validate_selection_run", "validate_study_metrics"
        )
        metrics = synthetic_metrics("sync")
        metrics["schedule_sha256"] = hashlib.sha256(
            json.dumps(units["study_schedule"](11), separators=(",", ":")).encode()
        ).hexdigest()
        metrics["config"] = units["study_config"]()
        metrics["counts"]["action_tokens"] -= 1
        receipt = {
            "run_id": metrics["run_id"],
            "run_order_index": 0,
            "run_passed": True,
            "manifest_sha256": "manifest",
            "source_sha256": "source",
            "model_sha256": "model",
            "config": metrics["config"],
            "checkpoint_receipts": metrics["checkpoint_receipts"],
            "arm": metrics["arm"],
            "seed": metrics["seed"],
            "updates": metrics["updates"],
            "counts": metrics["counts"],
            "data_sha256": metrics["data_sha256"],
            "schedule_sha256": metrics["schedule_sha256"],
            "runtime": metrics["runtime"],
            "model_config": metrics["model_config"],
            "trainable_parameters": metrics["trainable_parameters"],
            "training_policy": metrics["training_policy"],
            "runtime_policy": metrics["runtime_policy"],
        }
        with self.assertRaises(ValueError):
            units["validate_selection_run"](
                metrics,
                receipt,
                {"arm": "sync", "seed": 11, "index": 0},
                "manifest",
                "source",
                "model",
                metrics["data_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
