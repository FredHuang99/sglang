import os
import sys
import unittest
from unittest.mock import patch

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.registry import _get_config_info
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import FlexibleArgumentParser


def _from_dict_without_model_resolution(
    kwargs, pipeline_config: PipelineConfig | None = None
):
    pipeline_config = pipeline_config or QwenImagePipelineConfig()
    with patch.object(PipelineConfig, "from_kwargs", return_value=pipeline_config):
        return ServerArgs.from_dict(kwargs)


class TestServerArgsPathExpansion(unittest.TestCase):
    def _from_dict_without_model_resolution(self, kwargs):
        return _from_dict_without_model_resolution(kwargs)

    def test_tilde_model_path_is_expanded(self):
        args = self._from_dict_without_model_resolution(
            {"model_path": "~/fake/local/model"}
        )
        expected = os.path.expanduser("~/fake/local/model")
        self.assertEqual(args.model_path, expected)
        self.assertFalse(args.model_path.startswith("~"))

    def test_absolute_path_is_unchanged(self):
        args = self._from_dict_without_model_resolution(
            {"model_path": "/data/my-model"}
        )
        self.assertEqual(args.model_path, "/data/my-model")

    def test_component_paths_are_expanded_before_pipeline_resolution(self):
        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "component_paths": {"vae": "~/fake/local/vae"},
            }
        )

        self.assertEqual(
            args.component_paths["vae"], os.path.expanduser("~/fake/local/vae")
        )

    def test_profile_output_dir_is_expanded(self):
        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "profile_output_dir": "~/profile-output",
            }
        )
        self.assertEqual(
            args.profile_output_dir, os.path.expanduser("~/profile-output")
        )


class TestModelIdResolution(unittest.TestCase):
    def setUp(self):
        _get_config_info.cache_clear()

    def test_model_id_overrides_arbitrary_local_path(self):
        # a local path whose directory name does not match any HF repo name;
        # --model-id tells the engine which config to use
        info = _get_config_info("/data/my-custom-qwen", model_id="Qwen-Image")
        self.assertIsNotNone(info)

        self.assertIs(info.pipeline_config_cls, QwenImagePipelineConfig)

    def test_model_id_works_after_tilde_expansion(self):
        # simulate the full flow: user passes ~/..., engine expands and resolves
        expanded = os.path.expanduser("~/.cache/huggingface/hub/bbb/snapshots/ccc")
        _get_config_info.cache_clear()
        info = _get_config_info(expanded, model_id="Qwen-Image")
        self.assertIsNotNone(info)

    def test_model_id_unknown_falls_back_without_crash(self):
        # unrecognized model_id: should warn and fall back to path-based detection
        # with an unresolvable path, expect RuntimeError from the detector step
        with self.assertRaises((RuntimeError, Exception)):
            _get_config_info("/data/no-such-model", model_id="NonExistentModelXYZ")


class TestPerRoleParallelism(unittest.TestCase):
    """Test per-role parallelism args and get_role_parallelism helper."""

    def test_defaults_are_none(self):
        args = _from_dict_without_model_resolution({"model_path": "/fake"})
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        for role in [RoleType.ENCODER, RoleType.DENOISER, RoleType.DECODER]:
            par = args.get_role_parallelism(role)
            self.assertIsNone(par["tp_size"])
            self.assertIsNone(par["sp_degree"])
            self.assertIsNone(par["ulysses_degree"])
            self.assertIsNone(par["ring_degree"])

    def test_encoder_overrides(self):
        args = _from_dict_without_model_resolution(
            {
                "model_path": "/fake",
                "encoder_tp": 2,
            }
        )
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        par = args.get_role_parallelism(RoleType.ENCODER)
        self.assertEqual(par["tp_size"], 2)
        self.assertIsNone(par["sp_degree"])
        self.assertIsNone(par["ulysses_degree"])
        self.assertIsNone(par["ring_degree"])

    def test_denoiser_overrides(self):
        args = _from_dict_without_model_resolution(
            {
                "model_path": "/fake",
                "denoiser_tp": 1,
                "denoiser_sp": 8,
                "denoiser_ulysses": 4,
                "denoiser_ring": 2,
            }
        )
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        par = args.get_role_parallelism(RoleType.DENOISER)
        self.assertEqual(par["tp_size"], 1)
        self.assertEqual(par["sp_degree"], 8)
        self.assertEqual(par["ulysses_degree"], 4)
        self.assertEqual(par["ring_degree"], 2)

    def test_decoder_overrides(self):
        args = _from_dict_without_model_resolution(
            {
                "model_path": "/fake",
                "decoder_sp": 2,
            }
        )
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        par = args.get_role_parallelism(RoleType.DECODER)
        self.assertIsNone(par["tp_size"])
        self.assertEqual(par["sp_degree"], 2)
        self.assertIsNone(par["ulysses_degree"])
        self.assertIsNone(par["ring_degree"])

    def test_decoder_tp_is_alias_of_decoder_sp(self):
        args = _from_dict_without_model_resolution(
            {
                "model_path": "/fake",
                "decoder_tp": 2,
            }
        )
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        self.assertEqual(args.decoder_sp, 2)
        par = args.get_role_parallelism(RoleType.DECODER)
        self.assertIsNone(par["tp_size"])
        self.assertEqual(par["sp_degree"], 2)

    def test_conflicting_decoder_tp_and_decoder_sp_raise(self):
        with self.assertRaisesRegex(ValueError, "decoder_tp is deprecated"):
            _from_dict_without_model_resolution(
                {
                    "model_path": "/fake",
                    "decoder_tp": 2,
                    "decoder_sp": 4,
                }
            )

    def test_monolithic_returns_all_none(self):
        args = _from_dict_without_model_resolution(
            {
                "model_path": "/fake",
                "encoder_tp": 2,
            }
        )
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        par = args.get_role_parallelism(RoleType.MONOLITHIC)
        self.assertIsNone(par["tp_size"])
        self.assertIsNone(par["sp_degree"])

    def test_mixed_roles_independent(self):
        """Per-role args don't interfere with each other."""
        args = _from_dict_without_model_resolution(
            {
                "model_path": "/fake",
                "encoder_tp": 1,
                "denoiser_tp": 2,
                "decoder_sp": 4,
            }
        )
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        self.assertEqual(args.get_role_parallelism(RoleType.ENCODER)["tp_size"], 1)
        self.assertEqual(args.get_role_parallelism(RoleType.DENOISER)["tp_size"], 2)
        self.assertEqual(args.get_role_parallelism(RoleType.DECODER)["sp_degree"], 4)

    def test_cli_args_parsed(self):
        """Per-role parallelism args are parsed from CLI."""
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "/fake",
            "--denoiser-tp",
            "2",
            "--denoiser-sp",
            "4",
            "--denoiser-ulysses",
            "2",
            "--denoiser-ring",
            "2",
            "--encoder-tp",
            "1",
            "--decoder-sp",
            "8",
        ]
        args, unknown = parser.parse_known_args(argv)
        self.assertEqual(args.denoiser_tp, 2)
        self.assertEqual(args.denoiser_sp, 4)
        self.assertEqual(args.denoiser_ulysses, 2)
        self.assertEqual(args.denoiser_ring, 2)
        self.assertEqual(args.encoder_tp, 1)
        self.assertEqual(args.decoder_sp, 8)
        self.assertIsNone(args.decoder_tp)

    def test_profile_cli_args_parsed(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "/fake",
            "--profile-enabled",
            "true",
            "--request-profile-enabled",
            "true",
            "--profile-output-dir",
            "/tmp/profile",
            "--profile-run-id",
            "run-123",
        ]
        args, _unknown = parser.parse_known_args(argv)
        self.assertTrue(args.profile_enabled)
        self.assertTrue(args.request_profile_enabled)
        self.assertEqual(args.profile_output_dir, "/tmp/profile")
        self.assertEqual(args.profile_run_id, "run-123")
        self.assertEqual(args.diffusion_weight_staging, "none")
        self.assertEqual(args.diffusion_weight_load_mode, "default")

    def test_request_profile_cli_default_is_disabled(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        args, _unknown = parser.parse_known_args(["--model-path", "/fake"])
        self.assertFalse(args.request_profile_enabled)

    def test_profile_enabled_does_not_enable_request_profile_by_default(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        args, _unknown = parser.parse_known_args(
            ["--model-path", "/fake", "--profile-enabled", "true"]
        )
        self.assertTrue(args.profile_enabled)
        self.assertFalse(args.request_profile_enabled)

    def test_diffusion_weight_staging_cli_args_parsed(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        for mode in ("none", "pageable", "pinned", "auto"):
            args, _unknown = parser.parse_known_args(
                ["--model-path", "/fake", "--diffusion-weight-staging", mode]
            )
            self.assertEqual(args.diffusion_weight_staging, mode)

    def test_diffusion_weight_broadcast_cli_args_parsed(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        args, _unknown = parser.parse_known_args(
            [
                "--model-path",
                "/fake",
                "--diffusion-weight-load-mode",
                "rank0-broadcast",
                "--diffusion-weight-broadcast-components",
                "transformer,vae",
            ]
        )

        self.assertEqual(args.diffusion_weight_load_mode, "rank0-broadcast")
        self.assertEqual(
            args.diffusion_weight_broadcast_components,
            "transformer,vae",
        )

    def test_diffusion_weight_broadcast_components_normalized(self):
        server_args = _from_dict_without_model_resolution(
            {
                "model_path": "/fake",
                "diffusion_weight_load_mode": "rank0-broadcast",
                "diffusion_weight_broadcast_components": "transformer,vae",
            }
        )

        self.assertEqual(server_args.diffusion_weight_load_mode, "rank0-broadcast")
        self.assertEqual(
            server_args.diffusion_weight_broadcast_components,
            ["transformer", "vae"],
        )


class TestPipelineResolutionCliOverride(unittest.TestCase):
    def setUp(self):
        _get_config_info.cache_clear()

    def test_resolution_flag_overrides_qwen_image_layered_pipeline_config(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "Qwen/Qwen-Image-Layered",
            "--resolution",
            "768",
        ]

        with patch.object(sys, "argv", ["sglang"] + argv):
            args, unknown_args = parser.parse_known_args(argv)
            server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertEqual(server_args.pipeline_config.resolution, 768)


class TestDisaggTimeoutArgs(unittest.TestCase):
    def test_disagg_defaults_match_reviewed_values(self):
        args = _from_dict_without_model_resolution({"model_path": "/fake"})
        self.assertEqual(args.disagg_max_slots_per_instance, 8)
        self.assertEqual(args.disagg_downstream_wait_timeout, 1800)
        self.assertEqual(args.disagg_timeout, 3600)

    def test_downstream_wait_timeout_cli_arg_is_parsed(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "/fake",
            "--disagg-downstream-wait-timeout",
            "45",
        ]

        args, _unknown = parser.parse_known_args(argv)
        self.assertEqual(args.disagg_downstream_wait_timeout, 45)

    def test_disagg_timeout_help_uses_current_defaults(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        help_text = parser.format_help()

        self.assertIn("Default: 3600.", help_text)
        self.assertIn("Default: 1800.", help_text)

    def test_disagg_role_alias_cli_arg_is_accepted(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        args, _unknown = parser.parse_known_args(
            ["--model-path", "/fake", "--disagg-role", "denoising"]
        )

        self.assertEqual(args.disagg_role, "denoising")

    def test_disagg_role_alias_normalizes_to_denoiser(self):
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        args = _from_dict_without_model_resolution(
            {"model_path": "/fake", "disagg_role": "denoising"}
        )

        self.assertEqual(args.disagg_role, RoleType.DENOISER)


class TestDisaggTransferBackendArgs(unittest.TestCase):
    def test_transfer_backend_defaults_to_auto(self):
        args = _from_dict_without_model_resolution({"model_path": "/fake"})
        self.assertEqual(args.disagg_transfer_backend, "auto")

    def test_transfer_backend_cli_arg_is_parsed(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "/fake",
            "--disagg-transfer-backend",
            "mock",
        ]

        args, _unknown = parser.parse_known_args(argv)
        self.assertEqual(args.disagg_transfer_backend, "mock")


if __name__ == "__main__":
    unittest.main()
