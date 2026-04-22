# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CPU platform backend selection."""

import unittest

import torch

from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.cpu import CpuPlatform


class TestCpuPlatformAttentionBackend(unittest.TestCase):
    def test_cpu_defaults_to_torch_sdpa_backend(self):
        self.assertEqual(
            CpuPlatform.get_attn_backend_cls_str(None, 64, torch.float32),
            "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend",
        )

    def test_cpu_accepts_explicit_torch_sdpa_backend(self):
        self.assertEqual(
            CpuPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.TORCH_SDPA,
                64,
                torch.float32,
            ),
            "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend",
        )

    def test_cpu_rejects_flash_attention_backend(self):
        with self.assertRaisesRegex(ValueError, "FA is not supported on CPU"):
            CpuPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.FA,
                64,
                torch.float16,
            )


if __name__ == "__main__":
    unittest.main()
