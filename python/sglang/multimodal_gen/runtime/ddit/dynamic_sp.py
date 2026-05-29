"""Dynamic sequence-parallel process-group registry for DDiT."""

from __future__ import annotations

import contextlib
import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
    SequenceParallelGroupCoordinator,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_group
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .config import parse_allowed_gpu_counts, parse_local_ranks, resolve_ddit_sp_degrees

logger = init_logger(__name__)


@dataclass(frozen=True)
class DynamicSPGroupSpec:
    ranks: tuple[int, ...]
    ulysses_degree: int
    ring_degree: int


@dataclass(frozen=True)
class DynamicSPEnsureResult:
    spec: DynamicSPGroupSpec
    group: SequenceParallelGroupCoordinator | None
    created: bool


def _rank_groups_covering_world(active_ranks: tuple[int, ...]) -> list[list[int]]:
    world_size = dist.get_world_size()
    active = set(active_ranks)
    groups: list[list[int]] = [list(active_ranks)]
    groups.extend([rank] for rank in range(world_size) if rank not in active)
    return sorted(groups, key=lambda group: (group[0], len(group)))


def _subgroups_for_degrees(
    ranks: list[int],
    *,
    ulysses_degree: int,
    ring_degree: int,
) -> tuple[list[list[int]], list[list[int]]]:
    if len(ranks) == 1:
        return [ranks], [ranks]
    if ulysses_degree * ring_degree != len(ranks):
        raise ValueError(
            f"Invalid SP degrees for ranks {ranks}: "
            f"{ulysses_degree}x{ring_degree}"
        )
    ulysses_groups = []
    ring_groups = []
    for ring_idx in range(ring_degree):
        start = ring_idx * ulysses_degree
        ulysses_groups.append(ranks[start : start + ulysses_degree])
    for ulysses_idx in range(ulysses_degree):
        ring_groups.append(ranks[ulysses_idx::ulysses_degree])
    return ulysses_groups, ring_groups


class DynamicSPGroupRegistry:
    """Cache SequenceParallelGroupCoordinator objects by active rank tuple."""

    def __init__(self, server_args: Any):
        self.server_args = server_args
        self._cache: dict[DynamicSPGroupSpec, SequenceParallelGroupCoordinator] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def resolve_spec(self, ranks: tuple[int, ...]) -> DynamicSPGroupSpec:
        ranks = tuple(sorted(int(rank) for rank in ranks))
        ulysses_degree, ring_degree = resolve_ddit_sp_degrees(
            len(ranks),
            getattr(self.server_args, "ddit_sp_degree_map", None),
            server_args=self.server_args,
        )
        return DynamicSPGroupSpec(
            ranks=ranks,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
        )

    def has(self, ranks: tuple[int, ...]) -> bool:
        return self.resolve_spec(ranks) in self._cache

    def ensure(self, ranks: tuple[int, ...]) -> DynamicSPEnsureResult:
        spec = self.resolve_spec(ranks)
        if not dist.is_available() or not dist.is_initialized():
            return DynamicSPEnsureResult(spec=spec, group=None, created=False)
        group = self._cache.get(spec)
        if group is not None:
            return DynamicSPEnsureResult(spec=spec, group=group, created=False)
        group = self._build(spec)
        self._cache[spec] = group
        return DynamicSPEnsureResult(spec=spec, group=group, created=True)

    def _build(self, spec: DynamicSPGroupSpec) -> SequenceParallelGroupCoordinator:
        backend = current_platform.get_torch_distributed_backend_str()
        rank = dist.get_rank()
        group_ranks = _rank_groups_covering_world(spec.ranks)
        ulysses_pg = None
        ring_pg = None

        for group in group_ranks:
            if tuple(group) == spec.ranks:
                ulysses_degree = spec.ulysses_degree
                ring_degree = spec.ring_degree
            else:
                ulysses_degree = 1
                ring_degree = 1
            ulysses_groups, ring_groups = _subgroups_for_degrees(
                group,
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
            )
            for ranks in ulysses_groups:
                pg = dist.new_group(ranks, backend=backend)
                if rank in ranks:
                    ulysses_pg = pg
            for ranks in ring_groups:
                pg = dist.new_group(ranks, backend=backend)
                if rank in ranks:
                    ring_pg = pg

        if ulysses_pg is None or ring_pg is None:
            raise RuntimeError(f"Failed to build dynamic SP subgroup for rank {rank}")

        return SequenceParallelGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=get_world_group().local_rank,
            torch_distributed_backend=backend,
            group_name="ddit_sp_group",
            ulysses_group=ulysses_pg,
            ring_group=ring_pg,
        )

    def get(self, ranks: tuple[int, ...]) -> SequenceParallelGroupCoordinator | None:
        return self.ensure(ranks).group

    def prebuild(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_ranks = parse_local_ranks(
            getattr(self.server_args, "ddit_local_ranks", None), world_size
        )
        counts = parse_allowed_gpu_counts(
            getattr(self.server_args, "ddit_allowed_gpu_counts", None), len(local_ranks)
        )
        total_groups = sum(math.comb(len(local_ranks), count) for count in counts)
        built_groups = 0
        if rank == 0:
            logger.info(
                "DDiT dynamic SP prebuild start: role=%s, world_size=%s, "
                "local_ranks=%s, counts=%s, total_rank_tuples=%s, degree_map=%s",
                _ddit_role_value(self.server_args),
                world_size,
                local_ranks,
                counts,
                total_groups,
                getattr(self.server_args, "ddit_sp_degree_map", None),
            )
        for count in counts:
            if rank == 0:
                logger.info(
                    "DDiT dynamic SP prebuild count=%s start: combinations=%s",
                    count,
                    math.comb(len(local_ranks), count),
                )
            for ranks in itertools.combinations(local_ranks, count):
                built_groups += 1
                if rank == 0 and (
                    built_groups == 1
                    or built_groups == total_groups
                    or built_groups % 10 == 0
                ):
                    logger.info(
                        "DDiT dynamic SP prebuild progress: %s/%s latest_ranks=%s",
                        built_groups,
                        total_groups,
                        ranks,
                    )
                self.get(tuple(ranks))
            if rank == 0:
                logger.info("DDiT dynamic SP prebuild count=%s done", count)
        if rank == 0:
            logger.info("DDiT dynamic SP prebuild entering final barrier")
        dist.barrier()
        if rank == 0:
            logger.info(
                "DDiT dynamic SP prebuild done: cache_size=%s, "
                "total_rank_tuples=%s, allowed_counts=%s",
                self.cache_size,
                total_groups,
                counts,
            )

    @contextlib.contextmanager
    def use(self, ranks: tuple[int, ...]) -> Iterator[None]:
        if not dist.is_available() or not dist.is_initialized():
            yield
            return

        import sglang.multimodal_gen.runtime.distributed.parallel_groups as parallel_groups
        import sglang.multimodal_gen.runtime.distributed.parallel_state as parallel_state

        new_sp = self.get(tuple(ranks))
        old_sp = parallel_state._SP
        old_ulysses_pg = parallel_groups.PROCESS_GROUP.ULYSSES_PG
        old_ring_pg = parallel_groups.PROCESS_GROUP.RING_PG
        parallel_state._SP = new_sp
        parallel_groups.PROCESS_GROUP.ULYSSES_PG = new_sp.ulysses_group
        parallel_groups.PROCESS_GROUP.RING_PG = new_sp.ring_group
        try:
            yield
        finally:
            parallel_state._SP = old_sp
            parallel_groups.PROCESS_GROUP.ULYSSES_PG = old_ulysses_pg
            parallel_groups.PROCESS_GROUP.RING_PG = old_ring_pg


_REGISTRY_CACHE: dict[int, DynamicSPGroupRegistry] = {}


def get_dynamic_sp_registry(server_args: Any) -> DynamicSPGroupRegistry:
    key = id(server_args)
    registry = _REGISTRY_CACHE.get(key)
    if registry is None:
        registry = DynamicSPGroupRegistry(server_args)
        _REGISTRY_CACHE[key] = registry
    return registry


@contextlib.contextmanager
def use_dynamic_sp_group(server_args: Any, ranks: tuple[int, ...]) -> Iterator[None]:
    registry = get_dynamic_sp_registry(server_args)
    with registry.use(tuple(ranks)):
        yield


def prebuild_dynamic_sp_groups(server_args: Any) -> None:
    if should_prebuild_dynamic_sp_groups(server_args):
        get_dynamic_sp_registry(server_args).prebuild()


def _ddit_role_value(server_args: Any) -> str:
    role = getattr(server_args, "disagg_role", None)
    if role is None:
        return "monolithic"
    return getattr(role, "value", str(role)).lower()


def should_prebuild_dynamic_sp_groups(server_args: Any) -> bool:
    if not getattr(server_args, "enable_ddit", False):
        _log_prebuild_skip(server_args, "enable_ddit=false")
        return False
    if not getattr(server_args, "ddit_prebuild_sp_groups", True):
        _log_prebuild_skip(server_args, "disabled_by_arg")
        return False

    role_value = _ddit_role_value(server_args)
    if role_value in {"encoder", "server"}:
        _log_prebuild_skip(server_args, f"role={role_value}")
        return False
    return True


def _log_prebuild_skip(server_args: Any, reason: str) -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    logger.info(
        "Skipping DDiT dynamic SP prebuild: reason=%s, role=%s, "
        "enable_ddit=%s, ddit_prebuild_sp_groups=%s",
        reason,
        _ddit_role_value(server_args),
        getattr(server_args, "enable_ddit", False),
        getattr(server_args, "ddit_prebuild_sp_groups", True),
    )


def current_rank_in(ranks: tuple[int, ...]) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() in set(ranks)


def broadcast_tensor_from_rank(tensor: torch.Tensor, src: int) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    dist.broadcast(tensor, src=src, group=get_world_group().device_group)
    return tensor
