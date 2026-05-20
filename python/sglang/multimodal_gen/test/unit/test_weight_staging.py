# SPDX-License-Identifier: Apache-2.0

import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import load_file as safetensors_test_load_file
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
    _load_vae_state_dict_from_safetensors,
)
from sglang.multimodal_gen.runtime.loader.weight_staging import (
    maybe_stage_weight_iterator,
    normalize_weight_staging_mode,
    stage_weight_iterator,
)
from sglang.multimodal_gen.runtime.utils.weight_load_profiler import (
    WEIGHT_LOAD_PIN_MEMORY_ERROR,
    WEIGHT_LOAD_PIN_MEMORY_MS,
    WEIGHT_LOAD_PINNED_BYTES,
    WEIGHT_LOAD_PINNED_TENSOR_COUNT,
    WEIGHT_LOAD_READ_SAFETENSORS_MS,
    WEIGHT_LOAD_STAGED_TENSOR_COUNT,
    WEIGHT_LOAD_STAGING_EFFECTIVE,
    WEIGHT_LOAD_STAGING_REQUESTED,
    DiffusionWeightLoadProfiler,
)


class TestDiffusionWeightStaging(unittest.TestCase):
    def _profile(self):
        server_args = SimpleNamespace(
            profile_enabled=True,
            num_gpus=1,
            disagg_role_device="cuda",
        )
        return DiffusionWeightLoadProfiler(
            component="toy",
            server_args=server_args,
            enabled=True,
        )

    def test_normalize_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "Invalid diffusion weight staging"):
            normalize_weight_staging_mode("bad-mode")

    def test_pageable_staging_preserves_tensor_values(self):
        tensors = [
            ("a", torch.arange(4, dtype=torch.float32)),
            ("b", torch.ones((2, 3), dtype=torch.float16)),
        ]
        profile = self._profile()

        staged = list(
            stage_weight_iterator(
                tensors,
                staging_mode="pageable",
                weight_load_profile=profile,
            )
        )

        self.assertEqual([name for name, _ in staged], ["a", "b"])
        for (_, expected), (_, actual) in zip(tensors, staged, strict=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(actual, expected))

        record = profile.as_dict()
        self.assertEqual(record[WEIGHT_LOAD_STAGING_REQUESTED], "pageable")
        self.assertEqual(record[WEIGHT_LOAD_STAGING_EFFECTIVE], "pageable")
        self.assertEqual(record[WEIGHT_LOAD_STAGED_TENSOR_COUNT], 2)
        self.assertEqual(record[WEIGHT_LOAD_PINNED_TENSOR_COUNT], 0)

    def test_pinned_staging_records_pinned_tensors(self):
        tensors = [("a", torch.arange(8, dtype=torch.float32))]
        profile = self._profile()

        def fake_pin(tensor):
            time.sleep(0.001)
            return tensor.clone()

        staged = list(
            stage_weight_iterator(
                tensors,
                staging_mode="pinned",
                weight_load_profile=profile,
                pin_tensor=fake_pin,
            )
        )

        self.assertTrue(torch.equal(staged[0][1], tensors[0][1]))
        record = profile.as_dict()
        self.assertEqual(record[WEIGHT_LOAD_STAGING_REQUESTED], "pinned")
        self.assertEqual(record[WEIGHT_LOAD_STAGING_EFFECTIVE], "pinned")
        self.assertGreater(record[WEIGHT_LOAD_PIN_MEMORY_MS], 0.0)
        self.assertEqual(record[WEIGHT_LOAD_PINNED_TENSOR_COUNT], 1)
        self.assertEqual(
            record[WEIGHT_LOAD_PINNED_BYTES],
            tensors[0][1].numel() * tensors[0][1].element_size(),
        )
        self.assertIsNone(record[WEIGHT_LOAD_PIN_MEMORY_ERROR])

    def test_auto_staging_resolves_to_pageable_without_pin(self):
        tensors = [("a", torch.arange(4, dtype=torch.float32))]
        profile = self._profile()

        def fail_if_called(_tensor):
            raise AssertionError("auto should not try pinned staging")

        staged = list(
            stage_weight_iterator(
                tensors,
                staging_mode="auto",
                weight_load_profile=profile,
                pin_tensor=fail_if_called,
            )
        )

        self.assertTrue(torch.equal(staged[0][1], tensors[0][1]))
        record = profile.as_dict()
        self.assertEqual(record[WEIGHT_LOAD_STAGING_REQUESTED], "pageable")
        self.assertEqual(record[WEIGHT_LOAD_STAGING_EFFECTIVE], "pageable")
        self.assertEqual(record[WEIGHT_LOAD_PINNED_TENSOR_COUNT], 0)
        self.assertIsNone(record[WEIGHT_LOAD_PIN_MEMORY_ERROR])

    def test_non_rank0_skips_staging(self):
        tensors = [("a", torch.arange(4, dtype=torch.float32))]
        profile = self._profile()

        def fail_if_called(_tensor):
            raise AssertionError("pin should not be called on non-rank0")

        with mock.patch(
            "sglang.multimodal_gen.runtime.loader.weight_staging.is_weight_staging_rank0",
            return_value=False,
        ):
            staged = list(
                maybe_stage_weight_iterator(
                    tensors,
                    staging_mode="pinned",
                    weight_load_profile=profile,
                    pin_tensor=fail_if_called,
                )
            )

        self.assertTrue(torch.equal(staged[0][1], tensors[0][1]))
        record = profile.as_dict()
        self.assertEqual(record[WEIGHT_LOAD_STAGING_REQUESTED], "none")
        self.assertEqual(record[WEIGHT_LOAD_STAGED_TENSOR_COUNT], 0)

    def test_vae_staged_state_dict_preserves_load_state_dict_key_behavior(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            safetensors_path = f"{tmpdir}/model.safetensors"
            save_file(
                {
                    "weight": torch.ones((2, 2), dtype=torch.float32),
                    "extra": torch.ones((1,), dtype=torch.float32),
                },
                safetensors_path,
            )
            server_args = SimpleNamespace(
                diffusion_weight_staging="pageable",
                profile_enabled=True,
                num_gpus=1,
                disagg_role_device="cuda",
            )
            profile = DiffusionWeightLoadProfiler(
                component="vae",
                server_args=server_args,
                enabled=True,
            )

            with mock.patch(
                "sglang.multimodal_gen.runtime.loader.weight_staging.is_weight_staging_rank0",
                return_value=True,
            ), mock.patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader.safetensors_load_file",
                side_effect=safetensors_test_load_file,
            ) as load_file_mock:
                loaded = _load_vae_state_dict_from_safetensors(
                    [safetensors_path],
                    server_args,
                    profile,
                )
                load_file_mock.assert_called_once_with(safetensors_path)

        module = torch.nn.Linear(2, 2)
        incompatible = module.load_state_dict(loaded, strict=False)
        self.assertEqual(incompatible.missing_keys, ["bias"])
        self.assertEqual(incompatible.unexpected_keys, ["extra"])
        self.assertTrue(torch.equal(loaded["weight"], torch.ones((2, 2))))

        record = profile.as_dict()
        self.assertEqual(record[WEIGHT_LOAD_STAGING_REQUESTED], "pageable")
        self.assertEqual(record[WEIGHT_LOAD_STAGING_EFFECTIVE], "pageable")
        self.assertEqual(record[WEIGHT_LOAD_STAGED_TENSOR_COUNT], 2)
        self.assertGreater(record[WEIGHT_LOAD_READ_SAFETENSORS_MS], 0.0)


if __name__ == "__main__":
    unittest.main()
