# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Matrix formulation adapted from aiconfigurator.td_deployment_search.planner.
"""Sparse Bundle-Template MILP and explicit lexicographic objectives."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from .profiles import STAGES
from .templates import Node, Template, partitions

CAPACITY_TOL = 1e-8  # req/s; all solver capacity rows are scaled to req/min.


class SolveError(RuntimeError):
    def __init__(self, metadata: dict):
        self.metadata = metadata
        super().__init__(f"{metadata['phase']}: {metadata['status']} ({metadata['message']})")


class MILP:
    def __init__(self):
        self.upper: list[float] = []
        self.integer: list[int] = []
        self.rows: list[dict[int, float]] = []
        self.lower: list[float] = []
        self.bounds: list[float] = []
        self.history: list[dict] = []

    def variable(self, upper: float, integer: bool = True) -> int:
        self.upper.append(upper)
        self.integer.append(int(integer))
        return len(self.upper) - 1

    def constrain(
        self, row: dict[int, float], lower: float = -np.inf, upper: float = np.inf
    ) -> None:
        self.rows.append(row)
        self.lower.append(lower)
        self.bounds.append(upper)

    def solve(self, objective: dict[int, float], time_limit_s: float, phase: str) -> np.ndarray:
        if not math.isfinite(time_limit_s) or time_limit_s <= 0:
            raise ValueError("MILP time limit must be finite and positive")
        row_ids, col_ids, values = [], [], []
        for i, row in enumerate(self.rows):
            for j, value in row.items():
                if value:
                    row_ids.append(i)
                    col_ids.append(j)
                    values.append(value)
        a = coo_array((values, (row_ids, col_ids)), shape=(len(self.rows), len(self.upper))).tocsc()
        c = np.zeros(len(self.upper))
        for j, value in objective.items():
            c[j] = value
        started = time.perf_counter()
        result = milp(
            c,
            integrality=np.array(self.integer),
            bounds=Bounds(0, self.upper),
            constraints=LinearConstraint(a, self.lower, self.bounds),
            options={"time_limit": time_limit_s, "mip_rel_gap": 1e-9, "presolve": True},
        )
        names = {
            0: "optimal",
            1: "limit_reached",
            2: "infeasible",
            3: "unbounded",
            4: "solver_error",
        }

        def optional(name: str):
            value = getattr(result, name, None)
            return float(value) if value is not None and math.isfinite(value) else None

        metadata = {
            "phase": phase,
            "status": names.get(result.status, "unknown"),
            "status_code": int(result.status),
            "message": str(result.message),
            "elapsed_s": time.perf_counter() - started,
            "objective": optional("fun"),
            "mip_dual_bound": optional("mip_dual_bound"),
            "mip_gap": optional("mip_gap"),
            "time_limit_s": time_limit_s,
            "variables": len(self.upper),
            "integer_variables": sum(self.integer),
            "constraints": len(self.rows),
            "has_incumbent": result.x is not None,
        }
        self.history.append(metadata)
        if result.status != 0 or result.x is None:
            # A timeout incumbent is diagnostic only, never published as an optimal deployment.
            if result.x is not None:
                metadata["incumbent"] = result.x.tolist()
            raise SolveError(metadata)
        x = np.array(result.x)
        mask = np.array(self.integer, dtype=bool)
        if np.max(np.abs(x[mask] - np.rint(x[mask])), initial=0) > 1e-5:
            raise ValueError("Solver returned nonintegral deployment counts")
        x[mask] = np.rint(x[mask])
        if (
            not np.all(np.isfinite(x))
            or np.any(x < -1e-6)
            or np.any(x > np.array(self.upper) + 1e-6)
            or np.any(a @ x < np.array(self.lower) - 1e-6)
            or np.any(a @ x > np.array(self.bounds) + 1e-6)
        ):
            raise ValueError("Rounded MILP solution violates constraints")
        return x


def solve_counts(
    nodes: tuple[Node, ...], templates: dict[str, tuple[Template, ...]], time_limit_s: float
) -> tuple[list[tuple[Node, tuple[int, ...], Template]], dict]:
    model = MILP()
    node_groups = {hw: tuple(n for n in nodes if n.hardware == hw) for hw in templates}
    ys, zs = {}, {}
    for hw, group in node_groups.items():
        for partition in partitions(group[0].gpu_count):
            y = model.variable(len(group))
            ys[hw, partition] = y
            for width in sorted(set(partition), reverse=True):
                row = {y: -partition.count(width)}
                for template in templates[hw]:
                    if template.width == width:
                        z = model.variable(len(group) * partition.count(width))
                        zs[hw, partition, template.name] = (z, template)
                        row[z] = 1
                model.constrain(row, upper=0)
        model.constrain({idx: 1 for (h, _), idx in ys.items() if h == hw}, upper=len(group))
    lam = model.variable(np.inf, integer=False)
    for stage in STAGES:
        model.constrain({lam: 1, **{idx: -60 * t.caps[stage] for idx, t in zs.values()}}, upper=0)
    x = model.solve({lam: -1}, time_limit_s, "max_throughput")
    optimum = float(x[lam]) / 60
    model.constrain({lam: 1}, lower=max(0, optimum - CAPACITY_TOL) * 60)
    bundles = {idx: 1 for idx, _ in zs.values()}
    x = model.solve(bundles, time_limit_s, "min_instances_at_optimal_throughput")
    model.constrain(
        bundles, lower=round(sum(x[i] for i in bundles)), upper=round(sum(x[i] for i in bundles))
    )
    # An otherwise unused y variable must not activate an empty physical node.
    x = model.solve({idx: 1 for idx in ys.values()}, time_limit_s, "min_active_nodes")
    if optimum <= CAPACITY_TOL:
        raise ValueError("No positive-throughput deployment covers every required GPU stage")
    slots = defaultdict(list)
    cursor = Counter()
    for (hw, partition), idx in ys.items():
        for _ in range(int(x[idx])):
            node = node_groups[hw][cursor[hw]]
            cursor[hw] += 1
            offset = 0
            for width in partition:
                slots[hw, partition, width].append((node, tuple(range(offset, offset + width))))
                offset += width
    placements = []
    for (hw, partition, _), (idx, template) in zs.items():
        for _ in range(int(x[idx])):
            node, gpu_ids = slots[hw, partition, template.width].pop(0)
            placements.append((node, gpu_ids, template))
    selected_counts = Counter()
    for (hw, _, name), (idx, _) in zs.items():
        if x[idx] > 0:
            selected_counts[f"{hw}/{name}"] += int(x[idx])
    return placements, {
        "optimal_throughput_req_s": optimum,
        "capacity_tolerance_req_s": CAPACITY_TOL,
        "backend": "scipy.optimize.milp/HiGHS",
        "phases": model.history,
        "partition_variables": len(ys),
        "template_count_variables": len(zs),
        "selected_template_counts": dict(selected_counts),
    }
