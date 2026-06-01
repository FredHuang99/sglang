# SPDX-License-Identifier: Apache-2.0
"""Unit tests for disaggregation role-based module filtering."""

import unittest

from sglang.multimodal_gen.runtime.disaggregation.roles import (
    RoleType,
    filter_modules_for_role,
    get_module_role,
)


class TestRoleType(unittest.TestCase):
    def test_from_string(self):
        self.assertEqual(RoleType.from_string("monolithic"), RoleType.MONOLITHIC)
        self.assertEqual(RoleType.from_string("encoder"), RoleType.ENCODER)
        self.assertEqual(RoleType.from_string("denoiser"), RoleType.DENOISER)
        self.assertEqual(RoleType.from_string("decoder"), RoleType.DECODER)
        self.assertEqual(RoleType.from_string("ENCODER"), RoleType.ENCODER)

    def test_from_string_backward_compat(self):
        self.assertEqual(RoleType.from_string("denoising"), RoleType.DENOISER)
        self.assertEqual(RoleType.from_string("ddit_worker"), RoleType.DIT_VAE)

    def test_from_string_invalid(self):
        with self.assertRaises(ValueError):
            RoleType.from_string("invalid")

    def test_choices(self):
        choices = RoleType.choices()
        self.assertIn("monolithic", choices)
        self.assertIn("encoder", choices)
        self.assertIn("denoiser", choices)
        self.assertIn("denoising", choices)
        self.assertIn("decoder", choices)
        self.assertIn("dit_vae", choices)


class TestGetModuleRole(unittest.TestCase):
    def test_encoder_modules(self):
        self.assertEqual(get_module_role("text_encoder"), RoleType.ENCODER)
        self.assertEqual(get_module_role("text_encoder_2"), RoleType.ENCODER)
        self.assertEqual(get_module_role("tokenizer"), RoleType.ENCODER)
        self.assertEqual(get_module_role("tokenizer_2"), RoleType.ENCODER)
        self.assertEqual(get_module_role("image_encoder"), RoleType.ENCODER)
        self.assertEqual(get_module_role("image_processor"), RoleType.ENCODER)
        self.assertEqual(get_module_role("connectors"), RoleType.ENCODER)
        self.assertEqual(
            get_module_role("vision_language_encoder"), RoleType.ENCODER
        )
        self.assertEqual(get_module_role("hy3dshape_conditioner"), RoleType.ENCODER)
        self.assertEqual(
            get_module_role("hy3dshape_image_processor"), RoleType.ENCODER
        )

    def test_denoiser_modules(self):
        self.assertEqual(get_module_role("transformer"), RoleType.DENOISER)
        self.assertEqual(get_module_role("transformer_2"), RoleType.DENOISER)
        self.assertEqual(get_module_role("video_dit"), RoleType.DENOISER)
        self.assertEqual(get_module_role("video_dit_2"), RoleType.DENOISER)
        self.assertEqual(get_module_role("audio_dit"), RoleType.DENOISER)
        self.assertEqual(get_module_role("dual_tower_bridge"), RoleType.DENOISER)
        self.assertEqual(get_module_role("hy3dshape_model"), RoleType.DENOISER)

    def test_decoder_modules(self):
        self.assertEqual(get_module_role("vae"), RoleType.DECODER)
        self.assertEqual(get_module_role("audio_vae"), RoleType.DECODER)
        self.assertEqual(get_module_role("video_vae"), RoleType.DECODER)
        self.assertEqual(get_module_role("vocoder"), RoleType.DECODER)
        self.assertEqual(get_module_role("hy3dshape_vae"), RoleType.DECODER)

    def test_shared_modules(self):
        self.assertIsNone(get_module_role("scheduler"))
        self.assertIsNone(get_module_role("hy3dshape_scheduler"))


class TestFilterModulesForRole(unittest.TestCase):
    WAN_MODULES = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def test_monolithic_keeps_all(self):
        result = filter_modules_for_role(self.WAN_MODULES, RoleType.MONOLITHIC)
        self.assertEqual(result, self.WAN_MODULES)

    def test_encoder_does_not_keep_decoder_modules_by_default(self):
        result = filter_modules_for_role(self.WAN_MODULES, RoleType.ENCODER)
        self.assertEqual(result, ["text_encoder", "tokenizer", "scheduler"])

    def test_encoder_can_keep_explicit_cross_role_modules(self):
        result = filter_modules_for_role(
            self.WAN_MODULES,
            RoleType.ENCODER,
            extra_allowed_modules={"vae"},
        )
        self.assertEqual(result, ["text_encoder", "tokenizer", "vae", "scheduler"])

    def test_denoiser_skips_encoders_and_vae(self):
        result = filter_modules_for_role(self.WAN_MODULES, RoleType.DENOISER)
        self.assertEqual(result, ["transformer", "scheduler"])

    def test_decoder_keeps_vae_and_scheduler(self):
        result = filter_modules_for_role(self.WAN_MODULES, RoleType.DECODER)
        self.assertEqual(result, ["vae", "scheduler"])


class TestFilterModulesLTX2(unittest.TestCase):
    LTX2_MODULES = [
        "transformer",
        "text_encoder",
        "tokenizer",
        "scheduler",
        "vae",
        "audio_vae",
        "vocoder",
        "connectors",
    ]

    def test_decoder_includes_audio(self):
        result = filter_modules_for_role(self.LTX2_MODULES, RoleType.DECODER)
        self.assertEqual(result, ["scheduler", "vae", "audio_vae", "vocoder"])

    def test_encoder_does_not_keep_decoder_modules_by_default(self):
        result = filter_modules_for_role(self.LTX2_MODULES, RoleType.ENCODER)
        self.assertEqual(
            result, ["text_encoder", "tokenizer", "scheduler", "connectors"]
        )

    def test_denoiser_can_keep_ti2v_decoder_components(self):
        result = filter_modules_for_role(
            self.LTX2_MODULES,
            RoleType.DENOISER,
            extra_allowed_modules={"vae", "audio_vae"},
        )
        self.assertEqual(result, ["transformer", "scheduler", "vae", "audio_vae"])

    def test_dit_vae_keeps_denoiser_and_decoder_modules(self):
        result = filter_modules_for_role(self.LTX2_MODULES, RoleType.DIT_VAE)
        self.assertEqual(
            result,
            ["transformer", "scheduler", "vae", "audio_vae", "vocoder"],
        )


if __name__ == "__main__":
    unittest.main()
