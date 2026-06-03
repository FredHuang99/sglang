"""Microbenchmark Dynamic SP prebuild process-group cost.

Run with torchrun on a GPU machine. Example:

torchrun --nproc_per_node=8 examples/multimodal_gen/ddit_dynamic_sp_prebuild_bench.py \
  --mode lightweight --rank-tuples all --allowed-counts 1,2,4,8 \
  --model-id z-image --sp-degree-map shortpath --touch-collective \
  --out-dir /data/outputs/ddit_prebuild_bench/light_8
"""

from __future__ import annotations

import argparse
import csv
import inspect
import itertools
import json
import math
import os
import random
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


SHORTPATH_SP_DEGREE_TABLES: dict[str, dict[int, tuple[int, int]]] = {
    "wan2.1-t2v-1.3b": {1: (1, 1), 2: (2, 1), 4: (4, 1), 8: (2, 4)},
    "z-image": {1: (1, 1), 2: (2, 1), 4: (2, 2), 8: (2, 4)},
}


@dataclass
class BuildStats:
    created_device_groups: int = 0
    created_cpu_groups: int = 0
    reused_device_groups: int = 0
    new_group_ms: float = 0.0
    collective_touch_ms: float = 0.0
    elapsed_ms: float = 0.0
    error: str = ""


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def normalize_model_id(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if "wan" in normalized and "2.1" in normalized and "1.3" in normalized:
        return "wan2.1-t2v-1.3b"
    if "z-image" in normalized or "zimage" in normalized:
        return "z-image"
    return normalized


def resolve_sp_degrees(
    rank_count: int, *, model_id: str, degree_map: str
) -> tuple[int, int]:
    if degree_map == "shortpath":
        table = SHORTPATH_SP_DEGREE_TABLES[normalize_model_id(model_id)]
        return table[rank_count]
    parsed: dict[int, tuple[int, int]] = {}
    for item in degree_map.split(","):
        if not item.strip():
            continue
        count, pair = item.split(":")
        ulysses, ring = pair.lower().split("x")
        parsed[int(count)] = (int(ulysses), int(ring))
    return parsed.get(rank_count, (rank_count, 1))


def subgroups_for_degrees(
    ranks: tuple[int, ...], *, ulysses_degree: int, ring_degree: int
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    if len(ranks) == 1:
        return [ranks], [ranks]
    if ulysses_degree * ring_degree != len(ranks):
        raise ValueError(
            f"Invalid degree pair {ulysses_degree}x{ring_degree} for ranks={ranks}"
        )
    ulysses_groups = []
    ring_groups = []
    rank_list = list(ranks)
    for ring_idx in range(ring_degree):
        start = ring_idx * ulysses_degree
        ulysses_groups.append(tuple(rank_list[start : start + ulysses_degree]))
    for ulysses_idx in range(ulysses_degree):
        ring_groups.append(tuple(rank_list[ulysses_idx::ulysses_degree]))
    return ulysses_groups, ring_groups


def rank_tuples_for_args(args: argparse.Namespace, world_size: int) -> list[tuple[int, ...]]:
    ranks = tuple(range(world_size))
    counts = parse_csv_ints(args.allowed_counts)
    mode = args.rank_tuples
    if mode == "canonical":
        return [ranks[:count] for count in counts if count <= world_size]
    if mode.startswith("sample:"):
        sample_per_count = int(mode.split(":", 1)[1])
        rng = random.Random(args.seed)
        sampled: list[tuple[int, ...]] = []
        for count in counts:
            total = math.comb(world_size, count)
            if total <= sample_per_count:
                sampled.extend(
                    tuple(combo) for combo in itertools.combinations(ranks, count)
                )
            else:
                seen: set[tuple[int, ...]] = set()
                while len(seen) < sample_per_count:
                    seen.add(tuple(sorted(rng.sample(ranks, count))))
                sampled.extend(sorted(seen))
        return sampled
    all_tuples: list[tuple[int, ...]] = []
    for count in counts:
        all_tuples.extend(tuple(combo) for combo in itertools.combinations(ranks, count))
    if mode == "all":
        return all_tuples
    raise ValueError("--rank-tuples must be all, canonical, or sample:N")


def parse_rank_tuple_specs(value: str) -> tuple[tuple[int, ...], ...]:
    rank_tuples: list[tuple[int, ...]] = []
    for item in str(value or "").split(";"):
        item = item.strip()
        if not item:
            continue
        ranks = tuple(sorted(parse_csv_ints(item)))
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"Duplicate rank in --coverage-warmup-ranks item: {item}")
        rank_tuples.append(ranks)
    return tuple(rank_tuples)


def coverage_rank_tuples_for_args(
    args: argparse.Namespace, world_size: int
) -> tuple[tuple[int, ...], ...]:
    ranks = tuple(range(world_size))
    counts = parse_csv_ints(args.allowed_counts)
    mode = args.coverage_warmup_mode
    if mode == "custom":
        rank_tuples = parse_rank_tuple_specs(args.coverage_warmup_ranks)
    elif mode == "rank_count":
        rank_tuples = tuple(
            ranks[:count] for count in sorted(set(counts)) if 0 < count <= world_size
        )
    elif mode == "coverage":
        coverage: list[tuple[int, ...]] = []
        for count in sorted(set(counts)):
            if count <= 0 or count > world_size:
                continue
            if count == 1:
                coverage.extend((rank,) for rank in ranks)
                continue
            for start in range(0, world_size, count):
                group = ranks[start : start + count]
                if len(group) == count:
                    coverage.append(group)
        rank_tuples = tuple(coverage)
    else:
        raise ValueError("--coverage-warmup-mode must be coverage, rank_count, or custom")

    validated: list[tuple[int, ...]] = []
    rank_set = set(ranks)
    for rank_tuple in rank_tuples:
        normalized = tuple(sorted(int(rank) for rank in rank_tuple))
        if not normalized:
            continue
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Duplicate rank in coverage warmup tuple: {normalized}")
        if not set(normalized).issubset(rank_set):
            raise ValueError(
                f"Coverage warmup tuple {normalized} is outside world_size={world_size}"
            )
        validated.append(normalized)
    return tuple(sorted(set(validated), key=lambda item: (len(item), item)))


def new_group_optional_kwargs() -> dict[str, Any]:
    try:
        parameters = inspect.signature(dist.new_group).parameters
    except (TypeError, ValueError):
        return {}
    kwargs: dict[str, Any] = {}
    if "device_id" in parameters and torch.cuda.is_available():
        kwargs["device_id"] = torch.device(f"cuda:{torch.cuda.current_device()}")
    return kwargs


def memory_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "cpu_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        snapshot.update(
            {
                "cuda_allocated": torch.cuda.memory_allocated(device),
                "cuda_reserved": torch.cuda.memory_reserved(device),
                "cuda_max_allocated": torch.cuda.max_memory_allocated(device),
                "cuda_free": free,
                "cuda_total": total,
            }
        )
    return snapshot


class LightweightBuilder:
    def __init__(self, backend: str, touch_collective: bool):
        self.backend = backend
        self.touch_collective = touch_collective
        self.pg_cache: dict[tuple[int, ...], Any] = {}

    def build(self, ranks: tuple[int, ...], model_id: str, degree_map: str) -> BuildStats:
        stats = BuildStats()
        started = time.perf_counter()
        try:
            ulysses, ring = resolve_sp_degrees(
                len(ranks), model_id=model_id, degree_map=degree_map
            )
            subgroups_for_degrees(ranks, ulysses_degree=ulysses, ring_degree=ring)
            groups = [ranks]
            u_groups, r_groups = subgroups_for_degrees(
                ranks, ulysses_degree=ulysses, ring_degree=ring
            )
            groups.extend(u_groups)
            groups.extend(r_groups)
            for group_ranks in groups:
                self._get_or_create(group_ranks, stats)
        except Exception as exc:
            stats.error = f"{type(exc).__name__}: {exc}"
        stats.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return stats

    def _get_or_create(self, ranks: tuple[int, ...], stats: BuildStats) -> Any | None:
        ranks = tuple(sorted(int(rank) for rank in ranks))
        if len(ranks) <= 1:
            return None
        if ranks in self.pg_cache:
            stats.reused_device_groups += 1
            return self.pg_cache[ranks]
        if ranks == tuple(range(dist.get_world_size())):
            self.pg_cache[ranks] = dist.group.WORLD
            stats.reused_device_groups += 1
            return dist.group.WORLD
        started = time.perf_counter()
        pg = dist.new_group(ranks=list(ranks), backend=self.backend, **new_group_optional_kwargs())
        stats.new_group_ms += (time.perf_counter() - started) * 1000.0
        stats.created_device_groups += 1
        if self.touch_collective:
            stats.collective_touch_ms += touch_group(ranks, pg)
        self.pg_cache[ranks] = pg
        return pg


class HeavyProxyBuilder:
    """Approximate old heavy path with device+CPU PGs and optional SGLang coordinator."""

    def __init__(self, backend: str, touch_collective: bool, use_sglang: bool):
        self.backend = backend
        self.touch_collective = touch_collective
        self.use_sglang = use_sglang

    def build(self, ranks: tuple[int, ...], model_id: str, degree_map: str) -> BuildStats:
        stats = BuildStats()
        started = time.perf_counter()
        try:
            ulysses, ring = resolve_sp_degrees(
                len(ranks), model_id=model_id, degree_map=degree_map
            )
            u_groups, r_groups = subgroups_for_degrees(
                ranks, ulysses_degree=ulysses, ring_degree=ring
            )
            current_ulysses = self._create_member_or_singleton_pg(u_groups, stats)
            current_ring = self._create_member_or_singleton_pg(r_groups, stats)
            if self.use_sglang:
                self._create_sglang_sequence_coordinator(
                    ranks, current_ulysses, current_ring, stats
                )
            else:
                self._create_partition_device_and_cpu_groups(ranks, stats)
        except Exception as exc:
            stats.error = f"{type(exc).__name__}: {exc}"
        stats.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return stats

    def _create_member_or_singleton_pg(
        self, groups: list[tuple[int, ...]], stats: BuildStats
    ) -> Any:
        rank = dist.get_rank()
        selected_pg = None
        for group_ranks in groups:
            pg = self._create_device_pg(group_ranks, stats)
            if rank in group_ranks:
                selected_pg = pg
        if selected_pg is None:
            selected_pg = self._create_device_pg((rank,), stats)
        return selected_pg

    def _create_partition_device_and_cpu_groups(
        self, ranks: tuple[int, ...], stats: BuildStats
    ) -> None:
        active = tuple(sorted(ranks))
        partitions = [active] + [
            (rank,) for rank in range(dist.get_world_size()) if rank not in active
        ]
        for group_ranks in partitions:
            self._create_device_pg(group_ranks, stats)
            self._create_cpu_pg(group_ranks, stats)

    def _create_sglang_sequence_coordinator(
        self, ranks: tuple[int, ...], ulysses_pg: Any, ring_pg: Any, stats: BuildStats
    ) -> None:
        from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
            SequenceParallelGroupCoordinator,
        )

        active = list(sorted(ranks))
        group_ranks = [active] + [
            [rank] for rank in range(dist.get_world_size()) if rank not in set(active)
        ]
        SequenceParallelGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=int(os.environ.get("LOCAL_RANK", dist.get_rank())),
            torch_distributed_backend=self.backend,
            ulysses_group=ulysses_pg,
            ring_group=ring_pg,
            group_name="ddit_heavy_proxy",
        )
        # The coordinator internally creates device+CPU groups and PyNccl when possible.
        stats.created_device_groups += len(group_ranks)
        stats.created_cpu_groups += len(group_ranks)

    def _create_device_pg(self, ranks: tuple[int, ...], stats: BuildStats) -> Any:
        started = time.perf_counter()
        pg = dist.new_group(ranks=list(ranks), backend=self.backend, **new_group_optional_kwargs())
        stats.new_group_ms += (time.perf_counter() - started) * 1000.0
        stats.created_device_groups += 1
        if self.touch_collective and len(ranks) > 1:
            stats.collective_touch_ms += touch_group(ranks, pg)
        return pg

    def _create_cpu_pg(self, ranks: tuple[int, ...], stats: BuildStats) -> Any:
        started = time.perf_counter()
        pg = dist.new_group(ranks=list(ranks), backend="gloo")
        stats.new_group_ms += (time.perf_counter() - started) * 1000.0
        stats.created_cpu_groups += 1
        return pg


def touch_group(ranks: tuple[int, ...], pg: Any) -> float:
    started = time.perf_counter()
    rank = dist.get_rank()
    if torch.cuda.is_available() and rank in ranks:
        tensor = torch.ones((1,), device=torch.cuda.current_device(), dtype=torch.float32)
        dist.all_reduce(tensor, group=pg)
        torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - started) * 1000.0


def touch_cached_coverage_group(
    ranks: tuple[int, ...],
    pg_cache: dict[tuple[int, ...], Any],
) -> tuple[float, bool, str]:
    ranks = tuple(sorted(int(rank) for rank in ranks))
    if len(ranks) <= 1:
        if torch.cuda.is_available() and dist.get_rank() in ranks:
            torch.cuda.synchronize()
        return 0.0, False, "skip_singleton"
    if ranks == tuple(range(dist.get_world_size())):
        pg = dist.group.WORLD
    else:
        pg = pg_cache.get(ranks)
    if pg is None:
        return 0.0, False, "missing_process_group"
    return touch_group(ranks, pg), True, "touched"


def run_coverage_warmup(
    args: argparse.Namespace,
    builder: Any,
    world_size: int,
) -> dict[str, Any]:
    selected = coverage_rank_tuples_for_args(args, world_size)
    pg_cache = getattr(builder, "pg_cache", None)
    if pg_cache is None:
        raise RuntimeError("--coverage-warmup is only supported by lightweight mode")

    total_ms = 0.0
    touched = 0
    skipped_singleton = 0
    missing: list[tuple[int, ...]] = []
    per_tuple: list[dict[str, Any]] = []
    for ranks in selected:
        elapsed_ms, did_touch, status = touch_cached_coverage_group(ranks, pg_cache)
        total_ms += elapsed_ms
        if did_touch:
            touched += 1
        elif status == "skip_singleton":
            skipped_singleton += 1
        elif status == "missing_process_group":
            missing.append(ranks)
        per_tuple.append(
            {
                "ranks": list(ranks),
                "rank_count": len(ranks),
                "status": status,
                "collective_touch_ms": elapsed_ms,
            }
        )

    return {
        "coverage_warmup_enabled": True,
        "coverage_warmup_mode": args.coverage_warmup_mode,
        "coverage_warmup_kind": "collective_touch_only",
        "coverage_warmup_count": len(selected),
        "coverage_warmup_non_singleton_count": touched,
        "coverage_warmup_singleton_skipped_count": skipped_singleton,
        "coverage_warmup_missing_pg_count": len(missing),
        "coverage_warmup_missing_pg_tuples": [list(ranks) for ranks in missing],
        "coverage_warmup_tuples": [list(ranks) for ranks in selected],
        "coverage_warmup_per_tuple": per_tuple,
        "total_coverage_warmup_ms": total_ms,
    }


def init_dist(args: argparse.Namespace) -> str:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    backend = args.backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        kwargs: dict[str, Any] = {"backend": backend, "init_method": "env://"}
        if torch.cuda.is_available():
            kwargs["device_id"] = torch.device(f"cuda:{torch.cuda.current_device()}")
        dist.init_process_group(**kwargs)
    return backend


def cleanup_dist(run_barrier: bool) -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    if not (dist.is_available() and dist.is_initialized()):
        return
    if run_barrier:
        try:
            dist.barrier()
        except Exception:
            pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def write_rank_outputs(out_dir: Path, rank: int, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with (out_dir / f"rank{rank}_groups.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    with (out_dir / f"rank{rank}_groups.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (out_dir / f"rank{rank}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("lightweight", "heavy"), default="lightweight")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--rank-tuples", default="all", help="all, canonical, or sample:N")
    parser.add_argument("--allowed-counts", default="1,2,4,8")
    parser.add_argument("--model-id", default="z-image")
    parser.add_argument("--sp-degree-map", default="shortpath")
    parser.add_argument("--touch-collective", action="store_true")
    parser.add_argument("--coverage-warmup", action="store_true")
    parser.add_argument(
        "--coverage-warmup-mode",
        choices=("coverage", "rank_count", "custom"),
        default="coverage",
    )
    parser.add_argument(
        "--coverage-warmup-ranks",
        default="0;0,1;0,1,2,3;0,1,2,3,4,5,6,7",
        help="Semicolon-separated rank tuples used when --coverage-warmup-mode=custom",
    )
    parser.add_argument("--heavy-sglang-coordinator", action="store_true")
    parser.add_argument("--max-specs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    completed = False
    try:
        backend = init_dist(args)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        rank_tuples = rank_tuples_for_args(args, world_size)
        if args.max_specs > 0:
            rank_tuples = rank_tuples[: args.max_specs]
        builder: Any
        if args.mode == "lightweight":
            builder = LightweightBuilder(backend, args.touch_collective)
        else:
            if args.coverage_warmup:
                raise ValueError("--coverage-warmup is only supported with --mode lightweight")
            builder = HeavyProxyBuilder(
                backend, args.touch_collective, args.heavy_sglang_coordinator
            )

        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for idx, ranks in enumerate(rank_tuples, start=1):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            before = memory_snapshot()
            stats = builder.build(ranks, args.model_id, args.sp_degree_map)
            after = memory_snapshot()
            row = {
                "spec_index": idx,
                "mode": args.mode,
                "rank": rank,
                "world_size": world_size,
                "ranks": json.dumps(list(ranks)),
                "rank_count": len(ranks),
                "created_device_groups": stats.created_device_groups,
                "created_cpu_groups": stats.created_cpu_groups,
                "reused_device_groups": stats.reused_device_groups,
                "new_group_ms": stats.new_group_ms,
                "collective_touch_ms": stats.collective_touch_ms,
                "elapsed_ms": stats.elapsed_ms,
                "error": stats.error,
                "cuda_allocated_delta": after.get("cuda_allocated", 0)
                - before.get("cuda_allocated", 0),
                "cuda_reserved_delta": after.get("cuda_reserved", 0)
                - before.get("cuda_reserved", 0),
                "cuda_free_delta": after.get("cuda_free", 0) - before.get("cuda_free", 0),
                "cpu_maxrss_kb": after.get("cpu_maxrss_kb", 0),
            }
            rows.append(row)
            if stats.error:
                break
        dist.barrier()
        prebuild_elapsed_s = time.perf_counter() - started
        coverage_summary: dict[str, Any]
        if args.coverage_warmup:
            coverage_summary = run_coverage_warmup(args, builder, world_size)
        else:
            coverage_summary = {
                "coverage_warmup_enabled": False,
                "coverage_warmup_mode": args.coverage_warmup_mode,
                "coverage_warmup_kind": "collective_touch_only",
                "coverage_warmup_count": 0,
                "coverage_warmup_non_singleton_count": 0,
                "coverage_warmup_singleton_skipped_count": 0,
                "coverage_warmup_missing_pg_count": 0,
                "coverage_warmup_missing_pg_tuples": [],
                "coverage_warmup_tuples": [],
                "coverage_warmup_per_tuple": [],
                "total_coverage_warmup_ms": 0.0,
            }
        dist.barrier()
        total_elapsed_s = time.perf_counter() - started
        summary = {
            "mode": args.mode,
            "rank": rank,
            "world_size": world_size,
            "rank_specs": len(rank_tuples),
            "completed_specs": len(rows),
            "prebuild_elapsed_s": prebuild_elapsed_s,
            "elapsed_s": total_elapsed_s,
            "total_created_device_groups": sum(r["created_device_groups"] for r in rows),
            "total_created_cpu_groups": sum(r["created_cpu_groups"] for r in rows),
            "total_reused_device_groups": sum(r["reused_device_groups"] for r in rows),
            "total_new_group_ms": sum(r["new_group_ms"] for r in rows),
            "total_collective_touch_ms": sum(r["collective_touch_ms"] for r in rows),
            "total_prebuild_plus_coverage_warmup_ms": (
                prebuild_elapsed_s * 1000.0
                + float(coverage_summary["total_coverage_warmup_ms"])
            ),
            "errors": [r for r in rows if r["error"]],
            **coverage_summary,
        }
        write_rank_outputs(Path(args.out_dir), rank, rows, summary)
        if rank == 0:
            print(json.dumps(summary, indent=2, sort_keys=True))
        completed = True
    finally:
        cleanup_dist(run_barrier=completed)


if __name__ == "__main__":
    main()
