#!/usr/bin/env python3
"""Reproduce the 2A versus hybrid soft-window FA-KV comparison."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from future_aware_kv import FutureAwareKVAttention, RotaryEmbedding


@dataclass(frozen=True)
class Task:
    groups: int = 4
    members: int = 16
    member_vocab: int = 4096
    label_vocab: int = 4
    random_boundary_shift: bool = False

    def __post_init__(self) -> None:
        if self.random_boundary_shift and self.members < 2:
            raise ValueError("random boundary shifts require at least two members per group")
        if self.random_boundary_shift and self.label_vocab < self.groups + 1:
            raise ValueError("random boundary shifts require at least groups+1 labels")

    @property
    def body_length(self) -> int:
        return self.groups * (self.members + 1) + int(self.random_boundary_shift)

    @property
    def query_count(self) -> int:
        return self.groups * self.members

    @property
    def pad_token(self) -> int:
        return self.member_vocab + self.label_vocab

    @property
    def vocab_size(self) -> int:
        return self.pad_token + 1


@dataclass
class Batch:
    body: Tensor
    queries: Tensor
    targets: Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(self.body.to(device), self.queries.to(device), self.targets.to(device))


def _unique_ids(rows: int, count: int, vocabulary: int, rng: torch.Generator) -> Tensor:
    if count > vocabulary:
        raise ValueError("cannot draw more unique IDs than the vocabulary contains")
    random_priority = torch.rand(rows, vocabulary, generator=rng)
    return random_priority.topk(count, dim=1, largest=False).indices


def generate_batch(task: Task, batch_size: int, rng: torch.Generator) -> Batch:
    """Pack groups as ``members..., label`` and query every member at once."""
    members = _unique_ids(
        batch_size, task.query_count, task.member_vocab, rng
    ).reshape(batch_size, task.groups, task.members)
    label_count = task.groups + int(task.random_boundary_shift)
    labels = _unique_ids(batch_size, label_count, task.label_vocab, rng)

    grouped = torch.cat(
        (members, labels[:, : task.groups, None] + task.member_vocab), -1
    )
    body = grouped.reshape(batch_size, task.groups * (task.members + 1))
    queries = members.reshape(batch_size, task.query_count)
    targets = labels[:, : task.groups].repeat_interleave(task.members, dim=1)
    if task.random_boundary_shift:
        shift = torch.randint(task.members, (batch_size,), generator=rng)
        time = body.shape[1]
        source = (torch.arange(time)[None, :] + shift[:, None]) % time
        # With shift=0 the appended label owns no members. That is intentional
        # in this small experiment and keeps all member-starting phases valid.
        body = torch.cat(
            (body.gather(1, source), labels[:, -1, None] + task.member_vocab), dim=1
        )
        wrapped = torch.arange(task.members)[None, :] < shift[:, None]
        targets[:, : task.members] = torch.where(
            wrapped, labels[:, -1, None], targets[:, : task.members]
        )
    return Batch(body, queries, targets)


class MLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.up = nn.Linear(width, 2 * width)
        self.down = nn.Linear(2 * width, width)
        for layer in (self.up, self.down):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(x)))


class IndependentQueryAttention(nn.Module):
    """Causal body attention plus queries that see body+self, not each other."""

    def __init__(self, width: int, heads: int, head_dim: int):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        self.q = nn.Linear(width, inner, bias=False)
        self.k = nn.Linear(width, inner, bias=False)
        self.v = nn.Linear(width, inner, bias=False)
        self.output = nn.Linear(inner, width, bias=False)
        for layer in (self.q, self.k, self.v, self.output):
            nn.init.xavier_normal_(layer.weight)

    def _project(self, x: Tensor, layer: nn.Linear) -> Tensor:
        return layer(x).reshape(*x.shape[:2], self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self, body: Tensor, queries: Tensor, rotary: RotaryEmbedding | None = None
    ) -> tuple[Tensor, Tensor]:
        qb, kb, vb = (self._project(body, layer) for layer in (self.q, self.k, self.v))
        qz, kz, vz = (self._project(queries, layer) for layer in (self.q, self.k, self.v))
        scale = self.head_dim**-0.5
        time = body.shape[1]
        if rotary is not None:
            qb, kb = (
                x.transpose(1, 2) for x in rotary(
                    qb.transpose(1, 2), kb.transpose(1, 2)
                )
            )
            qz, kz = (
                x.transpose(1, 2) for x in rotary(
                    qz.transpose(1, 2), kz.transpose(1, 2), position=time
                )
            )
        causal = torch.ones(time, time, dtype=torch.bool, device=body.device).tril()
        body_probability = (qb @ kb.transpose(-1, -2) * scale).masked_fill(
            ~causal, -torch.inf
        ).softmax(-1)
        body_output = body_probability @ vb

        cross_score = qz @ kb.transpose(-1, -2) * scale
        self_score = (qz * kz).sum(-1, keepdim=True) * scale
        query_probability = torch.cat((cross_score, self_score), -1).softmax(-1)
        query_output = query_probability[..., :-1] @ vb + query_probability[..., -1:] * vz
        body_output = self.output(body_output.transpose(1, 2).flatten(-2))
        query_output = self.output(query_output.transpose(1, 2).flatten(-2))
        return body_output, query_output


class CausalBlock(nn.Module):
    def __init__(self, width: int, heads: int, head_dim: int):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = IndependentQueryAttention(width, heads, head_dim)
        self.mlp = MLP(width)

    def forward(
        self, body: Tensor, queries: Tensor, rotary: RotaryEmbedding | None = None
    ) -> tuple[Tensor, Tensor]:
        db, dq = self.attention(self.norm1(body), self.norm1(queries), rotary)
        body, queries = body + db, queries + dq
        return body + self.mlp(self.norm2(body)), queries + self.mlp(self.norm2(queries))


class MatchingModel(nn.Module):
    def __init__(
        self,
        architecture: str,
        task: Task,
        width: int = 64,
        heads: int = 2,
        head_dim: int = 32,
        gate_init: float = 0.5,
        attention_init: str = "copy",
        alibi_scale: float = 1 / 64,
        threshold_past_init_fraction: float = -0.05,
        threshold_future_init_fraction: float = 0.25,
        threshold_temperature: float = 4.0,
    ):
        super().__init__()
        if architecture not in ("2a", "fakv"):
            raise ValueError("architecture must be '2a' or 'fakv'")
        if attention_init not in ("copy", "independent"):
            raise ValueError("attention_init must be 'copy' or 'independent'")
        self.architecture, self.task = architecture, task
        self.rotary = RotaryEmbedding(head_dim) if architecture == "2a" else None
        self.embedding = nn.Embedding(task.vocab_size, width, padding_idx=task.pad_token)
        nn.init.normal_(self.embedding.weight, std=width**-0.5)
        with torch.no_grad():
            self.embedding.weight[task.pad_token].zero_()

        if architecture == "fakv":
            self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
            self.attention = FutureAwareKVAttention(
                width,
                heads,
                head_dim,
                gate_init=gate_init,
                alibi_scale=alibi_scale,
                threshold_past_init_fraction=threshold_past_init_fraction,
                threshold_future_init_fraction=threshold_future_init_fraction,
                threshold_reference_length=task.body_length + 1,
                threshold_temperature=threshold_temperature,
            )
            if attention_init == "copy":
                with torch.no_grad():
                    self.attention.kc0.weight.copy_(self.attention.qc.weight)
            self.mlp = MLP(width)
        else:
            self.blocks = nn.ModuleList(
                CausalBlock(width, heads, head_dim) for _ in range(2)
            )
            # Frozen baseline recipe: initialize retrieval Q and K equally in
            # both layers, then let them train independently.
            if attention_init == "copy":
                with torch.no_grad():
                    for block in self.blocks:
                        block.attention.k.weight.copy_(block.attention.q.weight)
        self.final_norm = nn.LayerNorm(width)

    def forward(self, body_tokens: Tensor, query_tokens: Tensor) -> Tensor:
        body = self.embedding(body_tokens)
        queries = self.embedding(query_tokens)
        if self.rotary is not None:
            self.rotary.prepare(body_tokens.shape[1] + 1, body)
        if self.architecture == "fakv":
            queries = queries + self.attention.forward_queries(
                self.norm1(body), self.norm1(queries)
            )
            queries = queries + self.mlp(self.norm2(queries))
        else:
            for block in self.blocks:
                body, queries = block(body, queries, self.rotary)
        queries = self.final_norm(queries)
        label_embeddings = self.embedding.weight[
            self.task.member_vocab : self.task.member_vocab + self.task.label_vocab
        ]
        return queries @ label_embeddings.T


def evaluate(model: nn.Module, batches: list[Batch]) -> dict[str, float]:
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            logits = model(batch.body, batch.queries)
            rows.append({
                "accuracy": float((logits.argmax(-1) == batch.targets).float().mean()),
                "loss": float(F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())),
            })
    model.train()
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def threshold_temperature(
    step: int, total_steps: int, start: float, end: float
) -> float:
    """Hold, cosine-anneal, then hold the soft-window temperature."""
    progress = step / max(1, total_steps)
    if progress <= 0.5:
        return start
    if progress >= 0.9:
        return end
    phase = (progress - 0.5) / 0.4
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * phase))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("2a", "fakv"), required=True)
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--alibi-scale", type=float, default=1 / 64)
    parser.add_argument("--threshold-past-init-fraction", type=float, default=-0.05)
    parser.add_argument("--threshold-future-init-fraction", type=float, default=0.25)
    parser.add_argument("--threshold-temperature-start", type=float, default=4.0)
    parser.add_argument("--threshold-temperature-end", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--test-batches", type=int, default=64)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gate-init", type=float, default=0.5)
    parser.add_argument(
        "--attention-init", choices=("copy", "independent"), default="copy"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = max(1, 512 // args.members)  # 2048 query targets/update.

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    task = Task(
        members=args.members,
        label_vocab=5,
        random_boundary_shift=True,
    )
    device = torch.device(args.device)
    model = MatchingModel(
        args.architecture,
        task,
        gate_init=args.gate_init,
        attention_init=args.attention_init,
        alibi_scale=args.alibi_scale,
        threshold_past_init_fraction=args.threshold_past_init_fraction,
        threshold_future_init_fraction=args.threshold_future_init_fraction,
        threshold_temperature=args.threshold_temperature_start,
    ).to(device)
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 0.002 if args.architecture == "2a" else 0.008
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    train_rng = torch.Generator().manual_seed(100_000 + args.seed)
    val_rng = torch.Generator().manual_seed(200_000 + args.seed)
    validation = [
        generate_batch(task, args.batch_size, val_rng).to(device)
        for _ in range(args.eval_batches)
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    curves: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        if args.architecture == "fakv":
            model.attention.threshold_temperature = threshold_temperature(
                step,
                args.steps,
                args.threshold_temperature_start,
                args.threshold_temperature_end,
            )
        lr = learning_rate * min(1.0, step / 50)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = generate_batch(task, args.batch_size, train_rng).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.body, batch.queries)
        loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, validation)
            row = {
                "step": step,
                "train_loss": float(loss.detach()),
                "val_accuracy": metrics["accuracy"],
                "val_loss": metrics["loss"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            if args.architecture == "fakv":
                row["threshold_temperature"] = model.attention.threshold_temperature
                if step >= math.ceil(0.9 * args.steps):
                    model.attention.hard_window = True
                    hard_metrics = evaluate(model, validation)
                    model.attention.hard_window = False
                    row["hard_val_accuracy"] = hard_metrics["accuracy"]
                    row["hard_val_loss"] = hard_metrics["loss"]
            curves.append(row)
            print(json.dumps(row), flush=True)

    test = None
    hard_test = None
    if not args.skip_test:
        test_rng = torch.Generator().manual_seed(300_000 + args.seed)
        test_data = [
            generate_batch(task, args.batch_size, test_rng).to(device)
            for _ in range(args.test_batches)
        ]
        test = evaluate(model, test_data)
        if args.architecture == "fakv":
            model.attention.hard_window = True
            hard_test = evaluate(model, test_data)
            model.attention.hard_window = False
    threshold_step = next((
        c["step"] for c, a, b in zip(curves, curves[1:], curves[2:])
        if min(c["val_accuracy"], a["val_accuracy"], b["val_accuracy"]) >= 0.98
    ), None)
    auc = sum(
        (b["step"] - a["step"]) * (a["val_accuracy"] + b["val_accuracy"]) / 2
        for a, b in zip(curves, curves[1:])
    ) / max(1, curves[-1]["step"] - curves[0]["step"])
    summary = {
        "args": vars(args) | {"output": str(args.output)},
        "task": asdict(task),
        "learning_rate": learning_rate,
        "parameters": sum(p.numel() for p in model.parameters()),
        "threshold_step": threshold_step,
        "validation_auc": auc,
        "final_validation": curves[-1],
        "test": test,
        "hard_test": hard_test,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if args.architecture == "fakv":
        summary["hard_final_validation"] = {
            "step": curves[-1]["step"],
            "val_accuracy": curves[-1]["hard_val_accuracy"],
            "val_loss": curves[-1]["hard_val_loss"],
        }
    if args.architecture == "fakv":
        summary["learned_gates"] = {
            "bypass_k": model.attention.gate_k.detach().cpu().tolist(),
            "bypass_v": model.attention.gate_v.detach().cpu().tolist(),
        }
        thresholds = model.attention.window_thresholds(task.body_length + 1)
        summary["learned_thresholds"] = {
            "past": thresholds[0].detach().cpu().tolist(),
            "future": thresholds[1].detach().cpu().tolist(),
            "hard_past": torch.floor(thresholds[0]).detach().cpu().tolist(),
            "hard_future": torch.floor(thresholds[1]).detach().cpu().tolist(),
            "final_temperature": model.attention.threshold_temperature,
        }
    (args.output / "curve.json").write_text(json.dumps(curves, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
