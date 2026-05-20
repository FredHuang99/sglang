# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import time
import unittest

import torch

from sglang.multimodal_gen.runtime.loader.weight_warm_pool import (
    WeightWarmPool,
    build_weight_warm_pool_key,
    get_global_weight_warm_pool,
    is_weight_warm_pool_enabled,
    make_weight_warm_pool_entry,
    normalize_weight_warm_pool_components,
    normalize_weight_warm_pool_mode,
    put_weight_warm_pool_entry,
)


class TestDiffusionWeightWarmPool(unittest.TestCase):
    def tearDown(self):
        get_global_weight_warm_pool().clear()

    def test_normalize_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "Invalid diffusion weight warm-pool"):
            normalize_weight_warm_pool_mode("pinned")

    def test_component_list_normalizes_commas_and_semicolons(self):
        self.assertEqual(
            normalize_weight_warm_pool_components("transformer; vae,,"),
            ["transformer", "vae"],
        )

    def test_component_gate(self):
        self.assertTrue(
            is_weight_warm_pool_enabled(
                mode="pageable",
                components="transformer,vae",
                component="transformer",
            )
        )
        self.assertFalse(
            is_weight_warm_pool_enabled(
                mode="disabled",
                components="transformer",
                component="transformer",
            )
        )

    def test_key_changes_when_file_fingerprint_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.safetensors")
            with open(path, "wb") as fp:
                fp.write(b"abc")
            key0 = build_weight_warm_pool_key(
                component="transformer",
                model_path=tmpdir,
                safetensors_files=[path],
                dtype=torch.bfloat16,
                component_class="ToyTransformer",
            )

            time.sleep(0.001)
            with open(path, "ab") as fp:
                fp.write(b"d")
            key1 = build_weight_warm_pool_key(
                component="transformer",
                model_path=tmpdir,
                safetensors_files=[path],
                dtype=torch.bfloat16,
                component_class="ToyTransformer",
            )

        self.assertNotEqual(key0, key1)

    def test_pool_hit_preserves_tensor_metadata_and_values(self):
        tensors = {
            "a": torch.arange(4, dtype=torch.float32),
            "b": torch.ones((2, 2), dtype=torch.float16),
        }
        entry = make_weight_warm_pool_entry(
            tensors,
            reverse_param_names_mapping={"a": "a"},
        )
        pool = WeightWarmPool()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.safetensors")
            with open(path, "wb") as fp:
                fp.write(b"abc")
            key = build_weight_warm_pool_key(
                component="transformer",
                model_path=tmpdir,
                safetensors_files=[path],
                dtype=torch.float32,
                component_class="Toy",
            )

        self.assertTrue(pool.put(key, entry))
        hit = pool.get(key)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(set(hit.tensors), {"a", "b"})
        self.assertEqual(hit.tensors["a"].shape, tensors["a"].shape)
        self.assertEqual(hit.tensors["b"].dtype, tensors["b"].dtype)
        self.assertTrue(torch.equal(hit.tensors["a"], tensors["a"]))
        self.assertEqual(hit.reverse_param_names_mapping, {"a": "a"})

    def test_global_pool_skips_entry_larger_than_limit(self):
        tensors = {"a": torch.ones((1024,), dtype=torch.float32)}
        entry = make_weight_warm_pool_entry(tensors)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.safetensors")
            with open(path, "wb") as fp:
                fp.write(b"abc")
            key = build_weight_warm_pool_key(
                component="vae",
                model_path=tmpdir,
                safetensors_files=[path],
                dtype=torch.float32,
                component_class="ToyVAE",
            )

        self.assertFalse(put_weight_warm_pool_entry(key, entry, max_gb=1e-9))
        self.assertIsNone(get_global_weight_warm_pool().get(key))


if __name__ == "__main__":
    unittest.main()
