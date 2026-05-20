# SPDX-License-Identifier: Apache-2.0

import importlib.util
import pathlib
import unittest


def _load_benchmark_module():
    repo_root = pathlib.Path(__file__).resolve().parents[5]
    path = repo_root / "scripts" / "benchmark_diffusion_cold_start.py"
    spec = importlib.util.spec_from_file_location("benchmark_diffusion_cold_start", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestDiffusionColdStartBenchmarkSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = _load_benchmark_module()

    def test_split_setups_rejects_unknown(self):
        with self.assertRaisesRegex(ValueError, "Unknown setups"):
            self.benchmark.split_setups("baseline,missing")

    def test_summary_handles_baseline_without_broadcast_fields(self):
        result = {
            "setup": "baseline",
            "run_id": "run0",
            "launch_wall_s": 12.3,
            "ready": True,
        }
        records = [
            {
                "component": "transformer",
                "sp_rank": "0",
                "weight_load:read_safetensors_ms": 1000.0,
                "weight_load:cpu_materialize_ms": 500.0,
                "weight_load:total_bytes": 1024**3,
            },
            {
                "component": "transformer",
                "sp_rank": "1",
                "weight_load:read_safetensors_ms": 2000.0,
                "weight_load:total_bytes": 1024**3,
            },
        ]

        summary = self.benchmark.summarize_run(result, records)

        self.assertEqual(summary["ready"], 1.0)
        self.assertEqual(summary["launch_wall_s"], 12.3)
        self.assertEqual(summary["transformer_rank0_read_s"], 1.0)
        self.assertEqual(summary["transformer_nonrank_avg_read_s"], 2.0)
        self.assertEqual(summary["transformer_nonrank_max_total_gib"], 1.0)

    def test_aggregate_by_setup_emits_avg_and_max_columns(self):
        columns, rows = self.benchmark.aggregate_by_setup(
            [
                ("pageable", {"launch_wall_s": 10.0, "ready": 1.0}),
                ("pageable", {"launch_wall_s": 12.0, "ready": 1.0}),
            ]
        )

        self.assertIn("launch_wall_s_avg", columns)
        self.assertIn("launch_wall_s_max", columns)
        self.assertEqual(rows[0]["setup"], "pageable")
        self.assertEqual(rows[0]["launch_wall_s_avg"], 11.0)
        self.assertEqual(rows[0]["launch_wall_s_max"], 12.0)


if __name__ == "__main__":
    unittest.main()
