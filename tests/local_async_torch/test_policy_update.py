"""CPU checks for the real local PPO head update, without loading a model."""

import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from examples.local_async.train import (
    MAX_GRAD_NORM,
    Action,
    Group,
    apply_policy_update,
    study_config,
)
from scripts import verify_head_quality


def oracle(head, groups):
    gradient = [[0.0 for _ in row] for row in head]
    loss = 0.0
    action_count = sum(len(group.actions) for group in groups)
    for group in groups:
        for index, action in enumerate(group.actions):
            count = 2 if index % 2 == 0 else 10
            hidden = action.hidden.tolist()
            logits = [sum(head[row][column] * hidden[column] for column in range(len(hidden))) for row in range(count)]
            maximum = max(logits)
            exps = [math.exp(value - maximum) for value in logits]
            probabilities = [value / sum(exps) for value in exps]
            logprob = logits[action.index] - maximum - math.log(sum(exps))
            ratio = math.exp(logprob - action.logprob)
            advantage = group.rewards[index // 2] - 0.5
            clipped = min(ratio * advantage, min(1.2, max(0.8, ratio)) * advantage)
            loss -= clipped / action_count
            active = (advantage > 0 and ratio <= 1.2) or (advantage < 0 and ratio >= 0.8)
            if active:
                for row in range(count):
                    dlogit = -advantage * ratio * (float(row == action.index) - probabilities[row]) / action_count
                    for column, value in enumerate(hidden):
                        gradient[row][column] += dlogit * value
    return loss, gradient


class PolicyUpdateTests(unittest.TestCase):
    def test_comparator_rejects_missing_runs_updates_and_checkpoints(self):
        def complete():
            return {
                f"{arm}-seed-{seed}": {
                    "updates": [
                        *({"step": step, "loss": 0.0} for step in range(1, 257)),
                        *({"step": step, "validation": {}} for step in (32, 128, 256)),
                    ]
                }
                for arm in ("clipped", "unclipped")
                for seed in (11, 23, 37, 51, 71)
            }

        verify_head_quality._validate_inventory(complete())
        missing_run = complete()
        missing_run.pop("clipped-seed-11")
        missing_update = complete()
        missing_update["clipped-seed-11"]["updates"].pop(0)
        missing_checkpoint = complete()
        missing_checkpoint["clipped-seed-11"]["updates"].pop()
        for runs in (missing_run, missing_update, missing_checkpoint):
            with self.assertRaises(ValueError):
                verify_head_quality._validate_inventory(runs)

    def test_comparator_config_survives_json_receipt_round_trip(self):
        self.assertEqual(json.loads(json.dumps(verify_head_quality.CONFIG)), verify_head_quality.CONFIG)

    def test_comparator_draws_both_tool_actions_before_answers(self):
        actions = [
            Action(torch.zeros(2), 0, 0.0, 0),
            Action(torch.zeros(2), 1, 0.0, 0),
            Action(torch.zeros(2), 2, 0.0, 0),
            Action(torch.zeros(2), 3, 0.0, 0),
        ]
        prompts = []

        def sample(prompt, *_args):
            prompts.append(prompt)
            return actions[len(prompts) - 1]

        with (
            patch.object(verify_head_quality.reference, "_prompts", return_value=("tool", ["answer0", "answer1"])),
            patch.object(verify_head_quality, "_sample", side_effect=sample),
        ):
            verify_head_quality._group(0, 0, 11, [(1, 2)], torch.zeros((10, 2)), {}, {})

        self.assertEqual(prompts, ["tool", "tool", "answer0", "answer1"])

    def test_loss_gradient_sgd_step_and_installed_rows_match_closed_form(self):
        initial = [[(row - 4) * 0.07 + column * 0.03 for column in range(3)] for row in range(10)]
        head = torch.tensor(initial, dtype=torch.float32, requires_grad=True)
        ratios = ([1.0, 1.3, 0.7, 0.9], [1.0, 0.7, 1.3, 1.1])
        groups = []
        for group_index, rewards in enumerate(([1.0, 0.0], [0.0, 1.0])):
            actions = []
            for index, ratio in enumerate(ratios[group_index]):
                count = 2 if index % 2 == 0 else 10
                hidden = torch.tensor([1.0, 0.2 * (index + 1), -0.1 * (group_index + 1)])
                action_index = (index + group_index) % count
                logits = torch.nn.functional.linear(hidden, head.detach()[:count])
                old_logprob = float(torch.log_softmax(logits, -1)[action_index]) - math.log(ratio)
                actions.append(Action(hidden, action_index, old_logprob, 0))
            groups.append(Group(str(group_index), 0, actions, list(rewards)))

        expected_loss, expected_gradient = oracle(initial, groups)
        expected_norm = math.sqrt(sum(value * value for row in expected_gradient for value in row))
        gradient_scale = min(1.0, MAX_GRAD_NORM / expected_norm)
        expected_head = [
            [value - 0.03 * gradient_scale * expected_gradient[row][column] for column, value in enumerate(values)]
            for row, values in enumerate(initial)
        ]
        ids = torch.tensor([11, 3, 7, 1, 9, 5, 0, 4, 8, 2])
        installed = torch.full((12, 3), 42.0)
        model = SimpleNamespace(lm_head=SimpleNamespace(weight=installed))
        optimizer = torch.optim.SGD([head], lr=0.03)

        loss, grad_norm = apply_policy_update(groups, head, optimizer, model, ids)

        self.assertAlmostEqual(loss, expected_loss, places=6)
        self.assertAlmostEqual(grad_norm, expected_norm, places=6)
        torch.testing.assert_close(head, torch.tensor(expected_head), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(installed[ids], head.detach())
        torch.testing.assert_close(installed[torch.tensor([6, 10])], torch.full((2, 3), 42.0))

    def test_large_gradient_is_clipped_before_optimizer_step(self):
        initial = [[(row - 4) * 0.07 + column * 0.03 for column in range(3)] for row in range(10)]
        head = torch.tensor(initial, dtype=torch.float32, requires_grad=True)
        hidden = torch.tensor([1.0, 0.5, -0.25])
        action_index = 0
        logits = torch.nn.functional.linear(hidden, head.detach()[:2])
        action = Action(hidden, action_index, float(torch.log_softmax(logits, -1)[action_index]), 0)
        groups = [Group("large", 0, [action, action], [1000.0, 0.0])]
        ids = torch.arange(10)
        model = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.zeros((10, 3))))
        optimizer = torch.optim.SGD([head], lr=0.03)
        before = head.detach().clone()
        _, raw_gradient = oracle(initial, groups)
        raw_norm = math.sqrt(sum(value * value for row in raw_gradient for value in row))
        scale = MAX_GRAD_NORM / raw_norm
        expected_head = torch.tensor(
            [
                [value - 0.03 * scale * raw_gradient[row][column] for column, value in enumerate(values)]
                for row, values in enumerate(initial)
            ]
        )

        _, preclip_norm = apply_policy_update(groups, head, optimizer, model, ids)

        self.assertGreater(preclip_norm, MAX_GRAD_NORM)
        applied_norm = float((before - head.detach()).norm() / optimizer.param_groups[0]["lr"])
        self.assertAlmostEqual(applied_norm, MAX_GRAD_NORM, places=5)
        torch.testing.assert_close(head, expected_head, atol=1e-6, rtol=1e-6)
        self.assertEqual(study_config()["max_grad_norm"], MAX_GRAD_NORM)

        unclipped = torch.tensor(initial, dtype=torch.float32, requires_grad=True)
        unclipped_optimizer = torch.optim.SGD([unclipped], lr=0.03)
        apply_policy_update(groups, unclipped, unclipped_optimizer, model, ids, max_grad_norm=None)
        self.assertGreater(float((before - unclipped.detach()).norm() / 0.03), MAX_GRAD_NORM)


if __name__ == "__main__":
    unittest.main()
