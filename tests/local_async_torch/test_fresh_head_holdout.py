import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts import verify_fresh_head_holdout as holdout


class FreshHeadHoldoutProtocolTest(unittest.TestCase):
    def test_frozen_pair_partition_is_complete_disjoint_and_stable(self):
        pairs = holdout.fresh_pairs()
        self.assertEqual(len(pairs), 256)
        self.assertEqual(len({tuple(pair) for pair in pairs}), 256)
        self.assertTrue(all(10 <= a <= b <= 99 for a, b in pairs))
        self.assertTrue(set(map(tuple, pairs)).isdisjoint((a, b) for a in range(10) for b in range(a, 10)))
        digest = hashlib.sha256(json.dumps(pairs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest, "b8a258e32a3d503a63e0d91c1e876db970a6af4520dc3c79b2dacd8452f2159c")

    def test_standalone_import_and_freeze_do_not_load_torch(self):
        root = Path(__file__).resolve().parents[2]
        script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts import verify_fresh_head_holdout as holdout
holdout._check_old_receipts = lambda *_: ({'training_sha256': 'frozen', 'selections': {}, 'source': {}}, {}, {})
holdout.freeze('selection', 'model', Path(sys.argv[2]) / 'frozen-output')
assert 'torch' not in sys.modules
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", script, str(root), directory],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn('"pairs": 256', result.stdout)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
        "Optional model runtime is absent",
    )
    def test_production_script_import_path_resolves_repo_modules(self):
        root = Path(__file__).resolve().parents[2]
        script_path = root / "scripts/verify_fresh_head_holdout.py"
        probe = """
import runpy
import sys
runpy.run_path(sys.argv[1], run_name='fresh_holdout_import_probe')
from examples.local_async import train
from scripts import verify_head_quality
assert train.__name__ == 'examples.local_async.train'
assert verify_head_quality.__name__ == 'scripts.verify_head_quality'
"""
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [sys.executable, "-c", probe, str(script_path)],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )

    def test_freeze_rejects_selection_identity_change_before_receipt_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            frozen = {"selection_sha256": "tampered"}
            path = output / "freeze.json"
            path.write_text(json.dumps(frozen), encoding="utf-8")
            digest = holdout.sha(path)
            (output / "freeze.sha256").write_text(digest, encoding="ascii")
            with self.assertRaisesRegex(ValueError, "selection identity changed"):
                holdout._validate_frozen(output, digest)

    def test_freeze_rejects_protocol_mutation_even_with_new_file_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            frozen = {
                "selection_sha256": holdout.SELECTION_SHA256,
                "protocol": {"pairs": [[10, 11]], "oracle_pair_indices": list(range(16))},
                "protocol_sha256": "wrong",
            }
            path = output / "freeze.json"
            path.write_text(json.dumps(frozen), encoding="utf-8")
            digest = holdout.sha(path)
            (output / "freeze.sha256").write_text(digest, encoding="ascii")
            with self.assertRaisesRegex(ValueError, "pair protocol was changed"):
                holdout._validate_frozen(output, digest)


if __name__ == "__main__":
    unittest.main()
