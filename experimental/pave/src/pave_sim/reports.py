"""Reports are reconstructed from request/attempt records, without executing a simulation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from pave_ilp.profiles import digest

from .metrics import LABELS, METRICS, allocations, summary
from .records import read_jsonl, write_csv, write_json
from .provenance import validate_contexts
from .timing import seconds, ticks

PAIRS = (("B", "A"), ("C", "B"), ("C", "A"), ("D", "C"), ("D", "A"), ("E", "D"), ("E", "A"))


def display(value, metric: str = "") -> str:
    if value is None:
        return "不可定义"
    if metric in ("slo5", "slo10"):
        return f"{100 * value:.2f}%"
    return f"{value:.6f}" if isinstance(value, (int, float)) else str(value)


def table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(cell(row.get(key, "")) for key, _ in columns) + " |" for row in rows),
    ])


def load_results(root: Path) -> tuple[list[dict], list[dict]]:
    results, failed, seen = [], [], set()
    for path in sorted((root / "runs").glob("*/run.json")):
        meta = json.loads(path.read_text("utf-8"))
        if meta.get("status") != "complete":
            failed.append({"path": str(path), "status": meta.get("status"), "error": meta.get("error")})
            continue
        if meta.get("schema_version") != 1:
            raise ValueError(f"Unsupported run schema in {path}")
        if not isinstance(meta.get("run_id"), str) or not meta["run_id"]:
            raise ValueError(f"Incomplete input provenance: missing run_id in {path}")
        if meta["run_id"] in seen:
            raise ValueError(f"Duplicate run_id {meta['run_id']} in {path}")
        seen.add(meta["run_id"])
        directory = path.parent
        results.append({
            **meta, "requests": read_jsonl(directory / "requests.jsonl"),
            "attempts": read_jsonl(directory / "attempts.jsonl"),
            "instances": read_jsonl(directory / "instances.jsonl"),
            "flips": json.loads((directory / "flips.json").read_text("utf-8")),
        })
    if not results:
        raise ValueError("No completed runs found under input-dir/runs")
    return results, failed


def matching_base(rows: list[dict], target: dict, strategy: str) -> dict | None:
    found = []
    target_context = target.get("comparison_context")
    for row in rows:
        if (row["custom_scheduler"] or row["case_id"] != target["case_id"]
                or row["rate_per_min"] != target["rate_per_min"] or row["strategy"] != strategy
                or row["seed"] != target["seed"]):
            continue
        context = row.get("comparison_context")
        if bool(context) != bool(target_context):
            raise ValueError("Incomplete comparison provenance in a comparison row")
        if context and context["shared_sha256"] != target_context["shared_sha256"]:
            raise ValueError(f"Conflicting input provenance for {row['run_id']} and {target.get('run_id', 'display slot')}")
        window = context["conditions"]["window_s"] if context else row["window_s"]
        target_window = target_context["conditions"]["window_s"] if target_context else target["window_s"]
        if strategy != "A" and window != target_window:
            continue
        if strategy in ("D", "E") and row["margin"] != target["margin"]:
            continue
        if strategy != "A":
            period = context["conditions"]["effective_monitor_period_s"] if context else row.get("effective_monitor_period_s", window)
            target_period = target_context["conditions"]["effective_monitor_period_s"] if target_context else target.get("effective_monitor_period_s", target_window)
            if period != target_period:
                raise ValueError(f"Conflicting effective_monitor_period_s for {row['run_id']} and {target.get('run_id', 'display slot')}")
        found.append(row)
    if len(found) > 1:
        raise ValueError(f"Ambiguous comparison baseline: {[row['run_id'] for row in found]}")
    return found[0] if found else None


def comparison_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    comparisons, missing = [], []
    for newer, older in PAIRS:
        for target in rows:
            if target["strategy"] != newer or target["custom_scheduler"]:
                continue
            base = matching_base(rows, target, older)
            if base is None:
                missing.append({"comparison": f"{newer} vs {older}", "target_run_id": target["run_id"], "reason": "baseline_not_available"})
                continue
            for metric in METRICS:
                before, after = base[metric], target[metric]
                change = after - before
                directional = -change if metric in ("p50_s", "p99_s") else change
                same = math.isclose(before, after, rel_tol=1e-10, abs_tol=1e-12)
                status = "unchanged" if same else "improved" if directional > 0 else "regressed"
                comparisons.append({
                    "case_id": target["case_id"], "comparison": f"{newer} vs {older}", "metric": metric,
                    "rate_per_min": target["rate_per_min"], "window_s": target["window_s"], "margin": target["margin"],
                    "baseline_run_id": base["run_id"], "target_run_id": target["run_id"],
                    "seed": target["seed"], "comparison_shared_sha256": target.get("comparison_context", {}).get("shared_sha256"),
                    "baseline_value": before, "target_value": after, "absolute_change": change,
                    "relative_improvement_percent": directional / before * 100 if before != 0 else None,
                    "percentage_point_change": change * 100 if metric in ("slo5", "slo10") else None,
                    "status": status,
                })
    return comparisons, missing


def ladder_rows(comparisons: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[row["case_id"], row["comparison"], row["metric"]].append(row)
    result = []
    for (case_id, comparison, metric), rows in sorted(grouped.items()):
        improved = [r for r in rows if r["status"] == "improved"]
        percentages = [r["relative_improvement_percent"] for r in improved if r["relative_improvement_percent"] is not None]
        result.append({
            "case_id": case_id, "comparison": comparison, "metric": metric, "pairs": len(rows),
            "improved": len(improved), "unchanged": sum(r["status"] == "unchanged" for r in rows),
            "regressed": sum(r["status"] == "regressed" for r in rows), "defined_improved_percentages": len(percentages),
            "min_percent": min(percentages) if percentages else None,
            "avg_percent": math.fsum(percentages) / len(percentages) if percentages else None,
            "max_percent": max(percentages) if percentages else None,
            **{
                f"{status}_setups": [
                    {
                        "rate_per_min": r["rate_per_min"], "window_s": r["window_s"],
                        "margin": r["margin"], "run_id": r["target_run_id"],
                    }
                    for r in rows if r["status"] == status
                ]
                for status in ("improved", "unchanged", "regressed")
            },
        })
    return result


def slot_and_winner_rows(rows: list[dict], settings: dict) -> tuple[list[dict], list[dict]]:
    slots, winners = [], []
    for case_id in sorted({row["case_id"] for row in rows}):
        case_rows = [row for row in rows if row["case_id"] == case_id and not row["custom_scheduler"]]
        if not case_rows:
            continue
        for rate in settings["request_rates_per_min"]:
            for window in settings["windows_s"]:
                for margin in settings["margins"]:
                    probe = {"case_id": case_id, "rate_per_min": rate, "window_s": window, "margin": margin, "seed": case_rows[0]["seed"]}
                    context = case_rows[0].get("comparison_context")
                    if context:
                        canonical = lambda value: seconds(ticks(value, positive=True)) if context["shared"]["simulation_semantics"]["version"] >= 2 else float(value)
                        matching_window = [r for r in case_rows if r["strategy"] != "A" and r["rate_per_min"] == rate and r["comparison_context"]["conditions"]["window_s"] == canonical(window)]
                        period = matching_window[0]["comparison_context"]["conditions"]["effective_monitor_period_s"] if matching_window else settings.get("monitor_period_s") or window
                        conditions = {"rate_per_min": float(rate), "window_s": canonical(window), "effective_monitor_period_s": canonical(period)}
                        probe["comparison_context"] = {**context, "conditions": conditions, "context_sha256": digest({"shared": context["shared"], "conditions": conditions})}
                    available = {}
                    for strategy in ("A", "B", "C", "D", "E"):
                        actual = matching_base(case_rows, probe, strategy)
                        if actual:
                            available[strategy] = actual
                            slots.append({**probe, "strategy": strategy, "actual_run_id": actual["run_id"], "actual_margin": actual["margin"], "actual_window_s": actual["window_s"], **{m: actual[m] for m in METRICS}})
                    if not available:
                        continue
                    row = {**probe, "available_strategies": sorted(available), "complete_ABCDE": len(available) == 5}
                    common = set(available)
                    for metric in METRICS:
                        best = (min if metric in ("p50_s", "p99_s") else max)(r[metric] for r in available.values())
                        best_names = [name for name, r in available.items() if math.isclose(r[metric], best, rel_tol=1e-10, abs_tol=1e-12)]
                        row[f"{metric}_best"] = best_names
                        common.intersection_update(best_names)
                    row["best_on_all_five"] = sorted(common)
                    winners.append(row)
    return slots, winners


def margin_rows(rows: list[dict]) -> list[dict]:
    result = []
    for target in rows:
        if target["custom_scheduler"] or target["strategy"] not in ("D", "E"):
            continue
        base = matching_base(rows, target, "C")
        if base:
            result.append({"case_id": target["case_id"], "strategy": target["strategy"], "rate_per_min": target["rate_per_min"], "window_s": target["window_s"], "margin": target["margin"], "baseline_run_id": base["run_id"], "target_run_id": target["run_id"], "without_margin_triggered": base["flip_triggered"], "with_margin_triggered": target["flip_triggered"], "triggered_difference": target["flip_triggered"] - base["flip_triggered"], "without_margin_completed": base["flip_completed"], "with_margin_completed": target["flip_completed"], "includes_launch_optimization": target["strategy"] == "E"})
    return result


def scheduling_comparison(rows: list[dict], allocation_by_run: dict[str, list[dict]]) -> list[dict]:
    result = []
    for target in rows:
        if target["strategy"] != "C" or target["custom_scheduler"]:
            continue
        base = matching_base(rows, target, "B")
        if base is None:
            continue
        left = {(r["stage"], r["placement_id"]): r for r in allocation_by_run[base["run_id"]] if r["stage"] in ("PE", "DiT")}
        right = {(r["stage"], r["placement_id"]): r for r in allocation_by_run[target["run_id"]] if r["stage"] in ("PE", "DiT")}
        for key in sorted(left.keys() | right.keys()):
            row = {"case_id": target["case_id"], "rate_per_min": target["rate_per_min"], "window_s": target["window_s"], "stage": key[0], "placement_id": key[1], "B_run_id": base["run_id"], "C_run_id": target["run_id"]}
            for metric in ("received", "completed", "executed_work"):
                for name, data in (("B", left), ("C", right)):
                    row[f"{name}_{metric}"] = data.get(key, {}).get(metric, 0)
                    row[f"{name}_{metric}_share"] = data.get(key, {}).get(f"{metric}_share", 0.0)
                row[f"{metric}_share_change_pp"] = (row[f"C_{metric}_share"] - row[f"B_{metric}_share"]) * 100
            result.append(row)
    return result


def render_case(case_id: str, rows: list[dict], ladders: list[dict], winners: list[dict]) -> str:
    raw = []
    for row in rows:
        if row["case_id"] == case_id:
            raw.append({"strategy": row["strategy"] + (f" / custom {row['scheduler']}" if row["custom_scheduler"] else ""), "rate": row["rate_per_min"], "window": row["window_s"], "margin": row["margin"], **{m: display(row[m], m) for m in METRICS}, "flips": row["flip_completed"], "run_id": row["run_id"]})
    columns = [("strategy", "策略"), ("rate", "req/min"), ("window", "窗口秒"), ("margin", "margin"), *((m, LABELS[m]) for m in METRICS), ("flips", "完成flip")]
    lines = [f"# {case_id}", "", "以下是独立运行的原始指标。A/B/C 的 margin 展示引用不增加样本数。", "", table(raw, columns), "", "## 固定 SLO 基准", ""]
    first = next(row for row in rows if row["case_id"] == case_id)
    for kind, baseline in first["slo_baselines"].items():
        detail = " + ".join(f"{stage} {entry['mean_service_s']:.6f}s" for stage, entry in baseline["stages"].items())
        lines.extend([f"{kind}：{detail} = **{baseline['latency_s']:.6f}s**。各阶段按实例数量平均，所有策略共用。", ""])
    lines.extend(["## Ladder", "", "改善百分比使用未舍入数值；min/avg/max 仅统计改善且分母非零的 setup。下列清单顺序为 req/min / window秒 / margin。", ""])
    for newer, older in PAIRS:
        pair = f"{newer} vs {older}"
        relevant = [r for r in ladders if r["case_id"] == case_id and r["comparison"] == pair]
        if not relevant:
            continue
        lines.extend([f"### {pair}", ""])
        for row in relevant:
            percentages = [
                "不可定义" if row[field] is None else f"{row[field]:.6f}%"
                for field in ("min_percent", "avg_percent", "max_percent")
            ]
            lines.extend([
                f"**{LABELS[row['metric']]}**：改善 {row['improved']}/{row['pairs']}，"
                f"持平 {row['unchanged']}，退化 {row['regressed']}；"
                f"改善 min/avg/max：{' / '.join(percentages)}。", "",
            ])
            for status, label in (("improved", "改善"), ("unchanged", "持平"), ("regressed", "退化")):
                setups = "; ".join(
                    f"{s['rate_per_min']:g}/{s['window_s']:g}/{s['margin']:g}"
                    for s in row[f"{status}_setups"]
                ) or "无"
                lines.extend([f"{label} setup：{setups}。", ""])
    best_rows = [{"rate": w["rate_per_min"], "window": w["window_s"], "margin": w["margin"], **{m: "/".join(w[f"{m}_best"]) for m in METRICS}, "all": "/".join(w["best_on_all_five"]) or "无", "complete": w["complete_ABCDE"]} for w in winners if w["case_id"] == case_id]
    lines.extend(["## 每项指标最优策略", "", "只在实际存在的策略中比较；完整列为 false 时属于部分矩阵。", "", table(best_rows, [("rate", "req/min"), ("window", "窗口"), ("margin", "margin"), *((m, LABELS[m]) for m in METRICS), ("all", "五项共同最优"), ("complete", "完整ABCDE")]), "", "SLO 百分点变化及全部改善/退化 setup 见 comparisons.csv。Margin flip 次数见 margin_flips.csv；PE/DiT 的 C vs B 分配占比见 scheduling_C_vs_B.csv。", ""])
    return "\n".join(lines)


def generate_reports(root: Path, output: Path, settings: dict) -> dict:
    results, failed = load_results(root)
    contexts = validate_contexts(results)
    rows = [{**summary(result), "comparison_context": contexts[result["run_id"]]} for result in results]
    baselines = {}
    for row in rows:
        if row["case_id"] in baselines and baselines[row["case_id"]] != row["slo_baselines"]:
            raise ValueError("Strategies of the same case must share fixed SLO baselines")
        baselines[row["case_id"]] = row["slo_baselines"]
    allocations_by_run = {result["run_id"]: allocations(result) for result in results}
    comparisons, missing = comparison_rows(rows)
    ladders = ladder_rows(comparisons)
    slots, winners = slot_and_winner_rows(rows, settings)
    datasets = {"summary": rows, "comparisons": comparisons, "ladder": ladders, "slots": slots, "winners": winners, "margin_flips": margin_rows(rows), "scheduling_C_vs_B": scheduling_comparison(rows, allocations_by_run), "allocations": [row for values in allocations_by_run.values() for row in values]}
    output.mkdir(parents=True, exist_ok=True)
    for name, records in datasets.items():
        write_json(output / f"{name}.json", {"schema_version": 1, "records": records})
        write_csv(output / f"{name}.csv", records)
    cases = sorted({row["case_id"] for row in rows})
    for case_id in cases:
        (output / f"{case_id}.md").write_text(render_case(case_id, rows, ladders, winners), encoding="utf-8")
    lines = ["# PAVE Simulation 实验汇总", "", f"从请求及执行记录重新计算了 {len(rows)} 次独立运行，涵盖 {len(cases)} 组部署。展示 slot 共 {len(slots)} 行，引用原始 run ID，不代表新增运行。", "", "吞吐按请求 makespan 计算；SLO 使用固定的阶段平均串行参考时延。传输成本为零，PE reprefill 和实例启动成本计入事件时间。", "", "- [完整原始指标](summary.csv)", "- [按setup对齐的大表](slots.csv)", "- [全部逐指标比较及百分点差](comparisons.csv)", "- [Margin flip次数](margin_flips.csv)", "- [C vs B实例分配](scheduling_C_vs_B.csv)", ""]
    lines.extend(f"- [{case_id}]({case_id}.md)" for case_id in cases)
    if failed or missing:
        lines.extend(["", f"存在 {len(failed)} 个未完成运行、{len(missing)} 项缺失基准的比较；未将其当作成功或零收益。详见 report_manifest.json。"])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "completed_runs": len(rows), "display_slots": len(slots), "cases": cases, "comparison_contexts": contexts, "failed_or_incomplete_runs": failed, "missing_comparisons": missing, "recomputed_from": ["requests.jsonl", "attempts.jsonl", "instances.jsonl", "flips.json", "run.json"]}
    write_json(output / "report_manifest.json", manifest)
    return manifest
