"""Command-wave planning primitives for concurrent single-node DDiT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DDiTOp:
    action: str
    request_id: str | None
    ranks: tuple[int, ...]
    stage: str
    step: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    op_id: str | None = None


@dataclass(frozen=True)
class CommandWave:
    wave_id: int
    ops: tuple[DDiTOp, ...]

    def commands_by_rank(self, world_size: int) -> dict[int, dict[str, Any]]:
        commands = {
            rank: {
                "action": "idle",
                "wave_id": self.wave_id,
                "op_id": f"wave{self.wave_id}:idle:{rank}",
                "ranks": (rank,),
                "stage": "idle",
            }
            for rank in range(world_size)
        }
        for index, op in enumerate(self.ops):
            op_id = op.op_id or f"wave{self.wave_id}:op{index}:{op.action}"
            command = {
                "action": op.action,
                "request_id": op.request_id,
                "ranks": op.ranks,
                "stage": op.stage,
                "step": op.step,
                "wave_id": self.wave_id,
                "op_id": op_id,
                **op.payload,
            }
            for rank in op.ranks:
                if commands[rank]["action"] != "idle":
                    raise ValueError(
                        f"Rank {rank} is assigned to multiple DDiT ops in wave "
                        f"{self.wave_id}"
                    )
                commands[rank] = command
        return commands


class CommandWaveBuilder:
    """Build a single wave of rank-disjoint operations."""

    def __init__(self, wave_id: int, world_size: int):
        self.wave_id = wave_id
        self.world_size = world_size
        self._used: set[int] = set()
        self._ops: list[DDiTOp] = []

    def can_add(self, ranks: tuple[int, ...]) -> bool:
        rank_set = set(int(rank) for rank in ranks)
        if not rank_set:
            return False
        if any(rank < 0 or rank >= self.world_size for rank in rank_set):
            return False
        return not (self._used & rank_set)

    def add(self, op: DDiTOp) -> bool:
        ranks = tuple(sorted(int(rank) for rank in op.ranks))
        if not self.can_add(ranks):
            return False
        self._used.update(ranks)
        self._ops.append(
            DDiTOp(
                action=op.action,
                request_id=op.request_id,
                ranks=ranks,
                stage=op.stage,
                step=op.step,
                payload=dict(op.payload),
                op_id=op.op_id,
            )
        )
        return True

    def build(self) -> CommandWave:
        return CommandWave(self.wave_id, tuple(self._ops))
