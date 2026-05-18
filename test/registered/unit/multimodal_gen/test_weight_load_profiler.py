import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except Exception:  # pragma: no cover - local minimal runtimes may miss deps.

    def register_cpu_ci(*args, **kwargs):
        return None

    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")

_IMPORT_ERROR = None
try:
    import torch
    from safetensors.torch import save_file

    from sglang.multimodal_gen.runtime.loader.weight_utils import (
        safetensors_weights_iterator,
    )
    from sglang.multimodal_gen.runtime.utils.weight_load_profiler import (
        WEIGHT_LOAD_CPU_MATERIALIZE_MS,
        WEIGHT_LOAD_DISCOVER_FILES_MS,
        WEIGHT_LOAD_H2D_OR_PARAM_COPY_MS,
        WEIGHT_LOAD_NCCL_BROADCAST_MS,
        WEIGHT_LOAD_PIN_MEMORY_MS,
        WEIGHT_LOAD_RANK0_WAIT_MS,
        WEIGHT_LOAD_READ_SAFETENSORS_MS,
        WEIGHT_LOAD_TOTAL_BYTES,
        DiffusionWeightLoadProfiler,
    )
except Exception as exc:  # pragma: no cover - skipped when deps are unavailable.
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"torch runtime unavailable: {_IMPORT_ERROR}")
class TestDiffusionWeightLoadProfiler(CustomTestCase):
    def _server_args(self, profile_output_dir: str):
        return SimpleNamespace(
            profile_enabled=True,
            profile_output_dir=profile_output_dir,
            profile_run_id="unit-weight-load",
            disagg_role="denoiser",
            disagg_instance_id=3,
            num_gpus=4,
            resolved_role_device=lambda: "cuda",
        )

    def test_profile_json_contains_phase1_fields_and_rank_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._server_args(tmpdir)
            with patch.dict(
                os.environ,
                {"RANK": "7", "LOCAL_RANK": "1", "WORLD_SIZE": "8"},
                clear=False,
            ):
                profiler = DiffusionWeightLoadProfiler.from_server_args(
                    args, "transformer"
                )
                with profiler.timing_scope(WEIGHT_LOAD_DISCOVER_FILES_MS):
                    pass
                with profiler.timing_scope(WEIGHT_LOAD_CPU_MATERIALIZE_MS):
                    pass
                profiler.add_tensor_bytes(torch.zeros(2, 3, dtype=torch.float16))
                output_path = profiler.finalize(status="success")

            self.assertIsNotNone(output_path)
            self.assertTrue(os.path.exists(output_path))
            with open(output_path, encoding="utf-8") as fp:
                record = json.load(fp)

            self.assertEqual(record["component"], "transformer")
            self.assertEqual(record["rank"], "7")
            self.assertEqual(record["physical_rank"], "1")
            self.assertEqual(record["world_size"], "8")
            self.assertEqual(record["sp_rank"], "unknown")
            self.assertEqual(record["tp_rank"], "unknown")
            self.assertEqual(record[WEIGHT_LOAD_TOTAL_BYTES], 12)
            for field in (
                WEIGHT_LOAD_DISCOVER_FILES_MS,
                WEIGHT_LOAD_READ_SAFETENSORS_MS,
                WEIGHT_LOAD_CPU_MATERIALIZE_MS,
                WEIGHT_LOAD_PIN_MEMORY_MS,
                WEIGHT_LOAD_H2D_OR_PARAM_COPY_MS,
                WEIGHT_LOAD_NCCL_BROADCAST_MS,
                WEIGHT_LOAD_RANK0_WAIT_MS,
            ):
                self.assertIn(field, record)
                self.assertGreaterEqual(record[field], 0.0)

    def test_tiny_safetensors_iterator_records_read_time_and_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "model.safetensors")
            tensors = {
                "linear.weight": torch.ones(2, 3, dtype=torch.float32),
                "linear.bias": torch.zeros(2, dtype=torch.float32),
            }
            save_file(tensors, ckpt)

            profiler = DiffusionWeightLoadProfiler.from_server_args(
                self._server_args(tmpdir), "vae"
            )
            items = list(
                profiler.profile_safetensors_iterator(
                    safetensors_weights_iterator(
                        [ckpt], use_runai_model_streamer=False
                    )
                )
            )

            self.assertEqual({name for name, _ in items}, set(tensors))
            expected_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
            self.assertEqual(
                profiler.as_dict()[WEIGHT_LOAD_TOTAL_BYTES], expected_bytes
            )
            self.assertGreaterEqual(
                profiler.as_dict()[WEIGHT_LOAD_READ_SAFETENSORS_MS], 0.0
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
