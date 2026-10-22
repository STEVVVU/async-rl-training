"""Bounded pretrained policy-gradient reference in an owned arithmetic tool environment."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from skyrl.train.staleness import GroupDecision, StalenessFence


@dataclass
class Action:
    hidden: torch.Tensor
    index: int
    logprob: float
    version: int


@dataclass
class Group:
    uid: str
    oldest: int
    actions: list[Action]
    rewards: list[float]


def tool(a, b, choice):
    if choice not in (0, 1):
        raise ValueError("Only owned sum and difference tools are available")
    return (a + b) % 10 if choice == 0 else abs(a - b)


async def train(model_path: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(15)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            use_safetensors=True,
        )
        .to("cuda")
        .eval()
    )
    model.requires_grad_(False)
    # Untie the head so head updates cannot silently change the frozen input embeddings.
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone(), requires_grad=False)
    model.config.tie_word_embeddings = False
    token_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(10)]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError("This constrained reference requires single-token decimal digits")
    ids = torch.tensor([item[0] for item in token_ids], device="cuda")
    initial = model.lm_head.weight[ids].float().clone()
    head = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.SGD([head], lr=0.005)
    version = 0
    fence = StalenessFence(max_age=1)
    queue = asyncio.Queue()
    tasks = []
    terminal = {}
    trained_uids = []
    curve = []
    losses = []
    mixed = []
    events_path = output / "events.jsonl"
    events = events_path.open("w", encoding="utf-8")

    def emit(kind, **data):
        events.write(json.dumps({"event": kind, "version": version, **data}) + "\n")
        events.flush()

    hidden_cache = {}

    def hidden(prompt):
        if prompt not in hidden_cache:
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                result = model.model(**encoded, use_cache=False).last_hidden_state[0, -1].float().clone()
            hidden_cache[prompt] = result.clone()
        return hidden_cache[prompt]

    def first_prompt(a, b):
        return f"Compute ({a}+{b}) modulo 10. Choose tool 0 for sum, 1 for difference. Tool:"

    def second_prompt(a, b, choice):
        return first_prompt(a, b) + f" {choice}\nTool result: {tool(a, b, choice)}\nAnswer digit:"

    def sample(prompt, count):
        h = hidden(prompt)
        with torch.no_grad():
            dist = torch.distributions.Categorical(logits=torch.nn.functional.linear(h, head[:count]))
            index = dist.sample()
        return Action(h, int(index), float(dist.log_prob(index)), version)

    async def rollout(uid, a, b, gate, ready):
        oldest = version
        first = [sample(first_prompt(a, b), 2) for _ in range(2)]
        emit("partial", uid=uid, oldest=oldest, tool_actions=[a.index for a in first])
        ready.set()
        await gate.wait()
        actions, rewards = [], []
        for action in first:
            observation = tool(a, b, action.index)
            answer = sample(second_prompt(a, b, action.index), 10)
            reward = float(answer.index == (a + b) % 10) + 0.25 * (action.index == 0)
            actions.extend([action, answer])
            rewards.append(reward)
            emit(
                "tool_turn",
                uid=uid,
                tool=action.index,
                observation=observation,
                answer=answer.index,
                reward=reward,
                policy_versions=[action.version, answer.version],
            )
            await asyncio.sleep(0)
        await queue.put(Group(uid, oldest, actions, rewards))
        emit("completed", uid=uid, oldest=oldest)

    async def launch(uid, a, b):
        gate, ready = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(rollout(uid, a, b, gate, ready))
        tasks.append(task)
        task.add_done_callback(lambda _: ready.set())
        await ready.wait()
        if task.done():
            await task
        return gate, task

    def evaluate():
        expected = []
        with torch.no_grad():
            for a, b in [(6, 7), (7, 8), (8, 9), (9, 6)]:
                tool_probs = torch.softmax(torch.nn.functional.linear(hidden(first_prompt(a, b)), head[:2]), -1)
                total = 0.0
                for choice in (0, 1):
                    answers = torch.softmax(torch.nn.functional.linear(hidden(second_prompt(a, b, choice)), head), -1)
                    total += float(tool_probs[choice] * (answers[(a + b) % 10] + 0.25 * (choice == 0)))
                expected.append(total)
        curve.append({"version": version, "heldout_expected_reward": sum(expected) / len(expected)})

    try:
        emit("loaded", model=str(model_path), trainable_parameters=head.numel(), backbone_frozen=True)
        evaluate()
        slow_gate, slow_task = await launch("straggler", 1, 2)
        carry = None
        for step in range(4):
            fast_count = 2 if carry is None else 1
            for j in range(fast_count):
                gate, task = await launch(f"fast-{step}-{j}", step + 1, j + 2)
                gate.set()
                await task
            next_gate, next_task = await launch(f"carry-{step}", step + 2, 3)
            batch = []
            while len(batch) < 2:
                group = await asyncio.wait_for(queue.get(), timeout=30)
                decision = fence.decide(group.uid, group.oldest, version)
                emit("fence", uid=group.uid, oldest=group.oldest, decision=decision.value)
                if decision == GroupDecision.DUPLICATE:
                    continue
                terminal[group.uid] = decision.value
                if decision == GroupDecision.STALE:
                    continue
                batch.append(group)
            emit("pause", in_flight=["straggler", f"carry-{step}"] if step < 2 else [f"carry-{step}"])
            terms = []
            for group in batch:
                trained_uids.append(group.uid)
                versions = sorted({a.version for a in group.actions})
                if len(versions) > 1:
                    mixed.append({"uid": group.uid, "versions": versions})
                for i, action in enumerate(group.actions):
                    count = 2 if i % 2 == 0 else 10
                    logprob = torch.log_softmax(torch.nn.functional.linear(action.hidden, head[:count]), -1)[
                        action.index
                    ]
                    ratio = torch.exp(logprob - action.logprob)
                    advantage = group.rewards[i // 2] - 0.5
                    terms.append(-torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage))
            loss = torch.stack(terms).mean()
            optimizer.zero_grad()
            loss.backward()
            if not torch.isfinite(head.grad).all():
                raise RuntimeError("Nonfinite policy gradient")
            grad_norm = float(head.grad.norm())
            optimizer.step()
            with torch.no_grad():
                model.lm_head.weight.index_copy_(0, ids, head.to(model.lm_head.weight.dtype))
            version += 1
            losses.append(float(loss.detach()))
            emit("update", loss=losses[-1], gradient_norm=grad_norm, trained=[g.uid for g in batch])
            evaluate()
            emit("resume", in_flight=f"carry-{step}")
            next_gate.set()
            await next_task
            carry = next_task
            if version == 2:
                slow_gate.set()
                await slow_task
        while not queue.empty():
            group = queue.get_nowait()
            terminal[group.uid] = "epoch_tail_filtered"
            emit("epoch_tail_filtered", uid=group.uid)
        delta = head.detach() - initial
        delta_norm = float(delta.norm())
        actual_delta = float((model.lm_head.weight[ids].float() - initial).norm())
        assert delta_norm > 0 and actual_delta > 0
        assert terminal["straggler"] == "stale"
        assert mixed and len(trained_uids) == len(set(trained_uids)) == 8
        assert len(terminal) == 10
        checkpoint = {
            "version": version,
            "terminal_uids": terminal,
            "trained_uids": trained_uids,
            "model_revision": model_path.name,
            "token_ids": ids.tolist(),
        }
        (output / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        restored = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
        assert set(restored["terminal_uids"]) == set(terminal)
        assert not [uid for uid in terminal if uid not in restored["terminal_uids"]]
        save_file(
            {
                "head_delta": delta.cpu(),
                "token_ids": ids.cpu(),
                "initial_head": initial.cpu(),
                "head_weights": head.detach().cpu(),
            },
            str(output / "policy-delta.safetensors"),
        )
        saved = load_file(str(output / "policy-delta.safetensors"), device="cuda")
        restored_head = saved["head_weights"]
        assert torch.equal(restored_head, head.detach())
        with torch.no_grad():
            head.copy_(restored_head)
        candidates = list(terminal) + ["resumed-fresh"]
        pending = [uid for uid in candidates if uid not in restored["terminal_uids"]]
        assert pending == ["resumed-fresh"]
        gate, task = await launch(pending[0], 4, 5)
        gate.set()
        await task
        resumed = queue.get_nowait()
        assert resumed.uid == "resumed-fresh" and resumed.oldest == restored["version"]
        assert fence.decide(resumed.uid, resumed.oldest, version) == GroupDecision.KEEP
        metrics = {
            "updates": version,
            "losses": losses,
            "parameter_delta_l2": delta_norm,
            "actual_model_head_delta_l2": actual_delta,
            "mixed_version_groups": mixed,
            "heldout_curve": curve,
            "trained_groups": len(trained_uids),
            "terminal_groups": terminal,
            "seed": 15,
            "scope": "Qwen3-0.6B frozen backbone, ten pretrained output rows, constrained actions",
            "native_distributed_backend": "not executed",
        }
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        emit(
            "checkpoint_reload",
            skipped_terminal=len(restored["terminal_uids"]),
            fresh_uid=resumed.uid,
            head_equal=True,
            optimizer="SGD without momentum has no persistent tensor state",
        )
        print(json.dumps(metrics, indent=2))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        events.close()


def run(model_path, output):
    asyncio.run(train(model_path, output))
