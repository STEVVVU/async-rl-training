"""Execute checkpoint call-site bodies with controlled IO, without importing Ray."""

import ast
import asyncio
import os
import unittest
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Set, Tuple
from unittest.mock import AsyncMock, Mock

from test_staleness import native_units

ROOT = Path(__file__).resolve().parents[2]


def train_method():
    tree = ast.parse((ROOT / "skyrl/train/fully_async_trainer.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FullyAsyncRayPPOTrainer")
    return next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "train")


def branch_driver(body, context=None):
    function = ast.AsyncFunctionDef(
        name="drive",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self"), ast.arg(arg="cur_generation_group_mini_batch")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[node for node in body if not isinstance(node, ast.Break)],
        decorator_list=[],
    )
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "asyncio": asyncio,
        "logger": SimpleNamespace(warning=lambda *args: None, info=lambda *args: None),
        "epoch": 0,
        "is_epoch_end": True,
    }
    namespace.update(context or {})
    exec(compile(tree, "native-checkpoint-branch", "exec"), namespace)
    return namespace["drive"]


class CheckpointBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def trainer(self, current_step):
        saved = []

        @asynccontextmanager
        async def deadline():
            yield

        trainer = SimpleNamespace(
            global_step=current_step,
            all_timings={},
            cfg=SimpleNamespace(trainer=SimpleNamespace(ckpt_interval=1, hf_save_interval=0)),
            _phase_gauge=SimpleNamespace(timed_phase=lambda *args: nullcontext()),
            _step_deadline=deadline,
            _staleness_manager=SimpleNamespace(on_rollout_filtered=AsyncMock()),
            async_train_dataloader=SimpleNamespace(mark_filtered_uids=AsyncMock()),
        )
        trainer.save_checkpoints = lambda global_step=None: saved.append(
            trainer.global_step if global_step is None else global_step
        )
        return trainer, saved

    async def test_exhaustion_saves_last_completed_step_with_or_without_partial_tail(self):
        branch = next(
            node
            for node in ast.walk(train_method())
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "epoch_exhausted"
        )
        drive = branch_driver(branch.body)
        for tail in ([], [SimpleNamespace(uid="tail")]):
            with self.subTest(tail=bool(tail)):
                trainer, saved = self.trainer(2)
                await drive(trainer, tail)
                self.assertEqual(saved, [1])
                self.assertEqual(trainer.global_step, 2)
                if tail:
                    trainer.async_train_dataloader.mark_filtered_uids.assert_awaited_once_with(["tail"])

    async def test_final_safety_net_saves_completed_step_including_zero_updates(self):
        branch = next(
            node
            for node in train_method().body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "self.cfg.trainer.ckpt_interval > 0"
        )
        drive = branch_driver(branch.body)
        for current in (1, 2, 5):
            trainer, saved = self.trainer(current)
            await drive(trainer, [])
            self.assertEqual(saved, [current - 1])
            self.assertEqual(trainer.global_step, current)

    async def test_periodic_post_update_save_keeps_current_step(self):
        branch = next(
            node
            for node in ast.walk(train_method())
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "self.cfg.trainer.ckpt_interval > 0"
            and any(isinstance(child, ast.Name) and child.id == "is_epoch_end" for child in ast.walk(node))
        )
        trainer, saved = self.trainer(2)
        await branch_driver(branch.body)(trainer, [])
        self.assertEqual(saved, [2])

    async def test_checkpoint_serialization_and_resume_align_completed_steps_uids_and_capacity(self):
        payloads, writes = {}, {}

        def open_file(path, mode):
            return nullcontext(SimpleNamespace(path=path, write=lambda value: writes.__setitem__(path, value)))

        namespace = {
            "Optional": Optional,
            "Tuple": Tuple,
            "Set": Set,
            "str": str,
            "os": os,
            "io": SimpleNamespace(makedirs=Mock(), open_file=open_file, exists=lambda path: path in payloads),
            "torch": SimpleNamespace(
                save=lambda value, file: payloads.__setitem__(file.path, value),
                load=lambda file, **kwargs: payloads[file.path],
            ),
            "logger": Mock(),
            "asdict": lambda cfg: {},
            "Timer": lambda *args: nullcontext(),
        }
        base_tree = ast.parse((ROOT / "skyrl/train/trainer.py").read_text(encoding="utf-8"))
        base = next(node for node in base_tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
        base.bases = []
        base.body = [
            node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == "save_checkpoints"
        ]
        exec(compile(ast.Module(body=[base], type_ignores=[]), "native-base-checkpoint", "exec"), namespace)
        base_class = namespace["RayPPOTrainer"]
        base_class.load_checkpoints = lambda self: (
            self.saved_step,
            os.path.join("checkpoint", f"global_step_{self.saved_step}"),
        )
        async_tree = ast.parse((ROOT / "skyrl/train/fully_async_trainer.py").read_text(encoding="utf-8"))
        cls = next(
            node
            for node in async_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FullyAsyncRayPPOTrainer"
        )
        cls.body = [
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name in {"save_checkpoints", "load_checkpoints"}
        ]
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "native-async-checkpoint", "exec"), namespace)
        units = native_units()
        for completed in (0, 1, 2):
            with self.subTest(completed=completed):
                trainer = namespace["FullyAsyncRayPPOTrainer"]()
                trainer.global_step = completed + 1
                trainer.epoch = 0
                trainer.cfg = SimpleNamespace(trainer=SimpleNamespace(ckpt_path="checkpoint"))
                trainer.has_critic = False
                trainer.dispatch, trainer.tokenizer, trainer.all_timings = Mock(), None, {}
                trainer.train_dataloader = SimpleNamespace(state_dict=lambda: {})
                trainer._cleanup_old_checkpoints = Mock()
                trained = {str(i) for i in range(completed * 2)}
                consumed = trained | {"tail"}
                trainer.async_train_dataloader = SimpleNamespace(
                    get_consumed_uids_list=lambda: list(consumed), get_filtered_uids_list=lambda: ["tail"]
                )
                folder = trainer.save_checkpoints(completed)
                self.assertEqual(payloads[os.path.join(folder, "trainer_state.pt")]["global_step"], completed)
                self.assertEqual(writes[os.path.join("checkpoint", "latest_ckpt_global_step.txt")], str(completed))
                self.assertEqual(trainer.global_step, completed + 1)
                normal_folder = trainer.save_checkpoints()
                self.assertEqual(
                    payloads[os.path.join(normal_folder, "trainer_state.pt")]["global_step"], completed + 1
                )
                trainer.saved_step = completed
                step, _, restored_consumed, restored_filtered, epoch = trainer.load_checkpoints()
                self.assertEqual((restored_consumed, restored_filtered, epoch), (consumed, {"tail"}, 0))
                manager = units._AsyncStalenessManager(4, 2, 1)
                manager.load_state_from_checkpoint(step + 1)
                self.assertEqual(manager._current_global_step, completed + 1)
                self.assertEqual(manager._stat.accepted, len(restored_consumed) - len(restored_filtered))
                self.assertEqual(manager._compute_capacity_unlocked(), 4)

    async def test_zero_update_checkpoint_restores_filtered_ledger_in_training_prologue(self):
        branch = next(
            node
            for node in ast.walk(train_method())
            if isinstance(node, ast.If)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and ast.unparse(node.body[0]).startswith("self.async_train_dataloader.load_state_from_checkpoint")
        )
        trainer = SimpleNamespace(
            global_step=0,
            mini_batch_size=2,
            num_steps_per_epoch=4,
            async_train_dataloader=Mock(),
            _staleness_manager=Mock(),
        )
        context = {
            "loaded_consumed_data_uids_set": {"filtered"},
            "loaded_filtered_data_uids_set": {"filtered"},
            "loaded_epoch": 0,
        }
        await branch_driver([branch], context)(trainer, [])
        trainer.async_train_dataloader.load_state_from_checkpoint.assert_called_once_with({"filtered"}, {"filtered"})
        trainer._staleness_manager.load_state_from_checkpoint.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
