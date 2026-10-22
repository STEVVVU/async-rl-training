"""Dependency-free fence tests and source-sliced native consumer unit tests.

These execute the actual selected method bodies without importing Ray/FSDP. They
are not an import/integration or end-to-end test of the native trainer.
"""

import ast
import asyncio
import typing
import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from skyrl.train.staleness import GroupDecision, StalenessFence


def native_units():
    path = Path(__file__).resolve().parents[2] / "skyrl/train/fully_async_trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name in {"GeneratedOutputGroup", "_RolloutStat", "_AsyncStalenessManager", "_AsyncDataloader"}:
                selected.append(node)
            elif node.name == "FullyAsyncRayPPOTrainer":
                node.bases = []
                node.body = [
                    n
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == "_collect_generation_mini_batch"
                ]
                selected.append(node)
    namespace = {
        **vars(typing),
        "asyncio": asyncio,
        "dataclass": dataclass,
        "StatefulDataLoader": object,
        "GeneratorOutput": dict,
        "BackgroundFailure": object,
        "GroupDecision": GroupDecision,
        "StalenessFence": StalenessFence,
        "tqdm": lambda **kwargs: Mock(),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class FenceTests(unittest.TestCase):
    def test_boundaries_and_terminal_duplicate(self):
        fence = StalenessFence(1)
        self.assertEqual(fence.decide("fresh", 2, 2), GroupDecision.KEEP)
        self.assertEqual(fence.decide("bound", 1, 2), GroupDecision.KEEP)
        self.assertEqual(fence.decide("old", 0, 2), GroupDecision.STALE)
        self.assertEqual(fence.decide("old", 2, 2), GroupDecision.DUPLICATE)
        self.assertEqual(fence.decide("fresh", 0, 10), GroupDecision.DUPLICATE)

    def test_invalid_boundary_values_do_not_poison_uid(self):
        for value in [-1, True, 0.1, "1"]:
            with self.assertRaises(ValueError):
                StalenessFence(value)
        fence = StalenessFence(0)
        for uid, oldest, current in [("", 0, 0), ("x", 2, 1), ("x", True, 1), ("x", -1, 1)]:
            with self.assertRaises(ValueError):
                fence.decide(uid, oldest, current)
        self.assertEqual(fence.decide("x", 0, 0), GroupDecision.KEEP)


class NativeBodyTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_boundary_keeps_partial_batch_at_exhaustion(self):
        units = native_units()
        group = units.GeneratedOutputGroup({}, "boundary", 1)
        trainer = SimpleNamespace(
            mini_batch_size=2,
            sample_full_batch=False,
            strict_staleness=True,
            global_step=2,
            _staleness_fence=StalenessFence(1),
            all_metrics={},
            all_timings={},
            _phase_gauge=SimpleNamespace(timed_phase=lambda *args: nullcontext()),
            _drain_next_group=AsyncMock(side_effect=[group, None]),
            _log_group_processing_error=Mock(),
        )
        kept, dropped, exhausted = await units.FullyAsyncRayPPOTrainer._collect_generation_mini_batch(
            trainer, asyncio.Queue(), asyncio.Event(), object()
        )
        self.assertEqual(kept, [group])
        self.assertEqual(dropped, [])
        self.assertTrue(exhausted)

    async def test_default_consumer_preserves_warning_only_behavior(self):
        units = native_units()
        group = units.GeneratedOutputGroup({}, "ancient", 0)
        queue = asyncio.Queue()
        queue.put_nowait(group)

        class Failure:
            async def guard(self, awaitable):
                return await awaitable

        trainer = SimpleNamespace(
            mini_batch_size=1,
            sample_full_batch=False,
            strict_staleness=False,
            global_step=100,
            all_metrics={},
            all_timings={},
            _phase_gauge=SimpleNamespace(timed_phase=lambda *args: nullcontext()),
        )
        kept, dropped, exhausted = await units.FullyAsyncRayPPOTrainer._collect_generation_mini_batch(
            trainer, queue, asyncio.Event(), Failure()
        )
        self.assertEqual(kept, [group])
        self.assertEqual(dropped, [])
        self.assertFalse(exhausted)

    async def test_stale_and_replayed_delivery_filter_once_then_exhaust(self):
        units = native_units()
        manager = units._AsyncStalenessManager(4, 2, 1)
        await manager.acquire_submission_slot()
        await manager.on_rollout_accepted()
        group = units.GeneratedOutputGroup({}, "old", 0)
        drain = AsyncMock(side_effect=[group, group, None])
        loader = SimpleNamespace(mark_filtered_uids=AsyncMock())
        trainer = SimpleNamespace(
            mini_batch_size=2,
            sample_full_batch=False,
            strict_staleness=True,
            global_step=2,
            _staleness_fence=StalenessFence(1),
            _staleness_manager=manager,
            async_train_dataloader=loader,
            all_metrics={},
            all_timings={},
            _phase_gauge=SimpleNamespace(timed_phase=lambda *args: nullcontext()),
            _drain_next_group=drain,
            _log_group_processing_error=Mock(),
        )
        kept, dropped, exhausted = await asyncio.wait_for(
            units.FullyAsyncRayPPOTrainer._collect_generation_mini_batch(
                trainer, asyncio.Queue(), asyncio.Event(), object()
            ),
            timeout=2,
        )
        self.assertEqual(kept, [])
        self.assertTrue(exhausted)
        loader.mark_filtered_uids.assert_awaited_once_with(["old"])
        self.assertEqual(manager._stat.accepted, 0)
        self.assertEqual(manager._stat.filtered, 1)
        self.assertEqual(manager._stat.submitted, 1)
        self.assertEqual(drain.await_count, 3)

    async def test_rejection_restores_capacity_and_conserves_submission(self):
        units = native_units()
        manager = units._AsyncStalenessManager(1, 1, 0)
        await manager.acquire_submission_slot()
        await manager.on_rollout_rejected()
        self.assertEqual(
            manager._stat.submitted, manager._stat.accepted + manager._stat.filtered + manager._stat.running
        )
        self.assertEqual(manager._compute_capacity_unlocked(), 1)
        await manager.validate_state_at_epoch_end(1)

    async def test_resume_skips_stale_and_trained_uids_but_counts_only_trained(self):
        units = native_units()

        class Loader:
            def __iter__(self):
                return iter([[{"uid": str(i)}] for i in range(4)])

            def __len__(self):
                return 4

            def state_dict(self):
                return {}

            def load_state_dict(self, state):
                pass

        loader = units._AsyncDataloader(Loader(), 2)
        await loader.mark_consumed_uids(["0", "1"])
        await loader.mark_filtered_uids(["2"])
        restored = units._AsyncDataloader(Loader(), 2)
        restored.load_state_from_checkpoint(set(loader.get_consumed_uids_list()), set(loader.get_filtered_uids_list()))
        self.assertEqual(restored.num_trained(), 2)
        self.assertEqual(await restored.get_next_non_consumed_data(), [{"uid": "3"}])
        self.assertIsNone(await restored.get_next_non_consumed_data())


if __name__ == "__main__":
    unittest.main()
