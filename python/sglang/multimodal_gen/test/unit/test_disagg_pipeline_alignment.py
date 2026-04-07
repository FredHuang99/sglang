# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pipeline-specific disagg role alignment."""

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan3d import (
    Hunyuan3D2PipelineConfig,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.disaggregation.roles import (
    RoleType,
    filter_modules_for_role,
)
from sglang.multimodal_gen.runtime.pipelines.flux_2 import Flux2Pipeline
from sglang.multimodal_gen.runtime.pipelines.glm_image import GlmImagePipeline
from sglang.multimodal_gen.runtime.pipelines.hunyuan3d_pipeline import (
    Hunyuan3D2Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import LTX2Pipeline
from sglang.multimodal_gen.runtime.pipelines.mova_pipeline import (
    MOVAPipeline,
    MOVAPipelineAlias,
)
from sglang.multimodal_gen.runtime.pipelines.qwen_image import (
    QwenImageEditPipeline,
    QwenImageLayeredPipeline,
)
from sglang.multimodal_gen.runtime.pipelines.wan_i2v_dmd_pipeline import (
    WanImageToVideoDmdPipeline,
)
from sglang.multimodal_gen.runtime.pipelines.wan_i2v_pipeline import (
    WanImageToVideoPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.helios_denoising import (
    HeliosChunkedDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.hunyuan3d_shape import (
    Hunyuan3DShapeBeforeDenoisingStage,
    Hunyuan3DShapeExportStage,
    Hunyuan3DShapeSaveStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.mova import (
    MOVADecodingStage,
    MOVADenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args


class TestPipelineSpecificExtraModules(unittest.TestCase):
    def _get_extra_modules(
        self, pipeline_cls, role: RoleType, task_name: str
    ) -> set[str]:
        pipeline = object.__new__(pipeline_cls)
        return pipeline._get_extra_allowed_modules_for_role(role, task_name)

    def test_flux_encoder_keeps_vae(self):
        extras = self._get_extra_modules(Flux2Pipeline, RoleType.ENCODER, "ti2i")
        filtered = filter_modules_for_role(
            Flux2Pipeline._required_config_modules,
            RoleType.ENCODER,
            extra_allowed_modules=extras,
        )
        self.assertEqual(extras, {"vae"})
        self.assertEqual(
            set(filtered), {"text_encoder", "tokenizer", "vae", "scheduler"}
        )

    def test_qwen_image_edit_encoder_keeps_vae(self):
        extras = self._get_extra_modules(
            QwenImageEditPipeline, RoleType.ENCODER, "ti2i"
        )
        filtered = filter_modules_for_role(
            QwenImageEditPipeline._required_config_modules,
            RoleType.ENCODER,
            extra_allowed_modules=extras,
        )
        self.assertEqual(extras, {"vae"})
        self.assertEqual(
            set(filtered),
            {"processor", "scheduler", "text_encoder", "tokenizer", "vae"},
        )

    def test_qwen_image_layered_encoder_keeps_required_cross_role_modules(self):
        extras = self._get_extra_modules(
            QwenImageLayeredPipeline, RoleType.ENCODER, "ti2i"
        )
        filtered = filter_modules_for_role(
            QwenImageLayeredPipeline._required_config_modules,
            RoleType.ENCODER,
            extra_allowed_modules=extras,
        )
        self.assertEqual(extras, {"vae", "transformer"})
        self.assertIn("text_encoder", QwenImageLayeredPipeline._required_config_modules)
        self.assertEqual(
            set(filtered),
            {
                "text_encoder",
                "vae",
                "tokenizer",
                "processor",
                "transformer",
                "scheduler",
            },
        )

    def test_glm_image_encoder_keeps_vae_and_transformer(self):
        extras = self._get_extra_modules(GlmImagePipeline, RoleType.ENCODER, "ti2i")
        filtered = filter_modules_for_role(
            GlmImagePipeline._required_config_modules,
            RoleType.ENCODER,
            extra_allowed_modules=extras,
        )
        self.assertEqual(extras, {"vae", "transformer"})
        self.assertEqual(
            set(filtered),
            {
                "text_encoder",
                "tokenizer",
                "vae",
                "vision_language_encoder",
                "processor",
                "transformer",
                "scheduler",
            },
        )

    def test_wan_ti2v_denoiser_keeps_vae(self):
        for pipeline_cls in (WanImageToVideoPipeline, WanImageToVideoDmdPipeline):
            extras = self._get_extra_modules(pipeline_cls, RoleType.DENOISER, "ti2v")
            filtered = filter_modules_for_role(
                pipeline_cls._required_config_modules,
                RoleType.DENOISER,
                extra_allowed_modules=extras,
            )
            self.assertEqual(extras, {"vae"})
            self.assertEqual(set(filtered), {"vae", "transformer", "scheduler"})

    def test_ltx2_encoder_does_not_keep_decoder_modules(self):
        extras = self._get_extra_modules(LTX2Pipeline, RoleType.ENCODER, "ti2v")
        filtered = filter_modules_for_role(
            LTX2Pipeline._required_config_modules,
            RoleType.ENCODER,
            extra_allowed_modules=extras,
        )
        self.assertEqual(extras, set())
        self.assertEqual(
            set(filtered),
            {"text_encoder", "tokenizer", "scheduler", "connectors"},
        )

    def test_ltx2_ti2v_denoiser_keeps_vae_and_audio_vae(self):
        extras = self._get_extra_modules(LTX2Pipeline, RoleType.DENOISER, "ti2v")
        filtered = filter_modules_for_role(
            LTX2Pipeline._required_config_modules,
            RoleType.DENOISER,
            extra_allowed_modules=extras,
        )
        self.assertEqual(extras, {"vae", "audio_vae"})
        self.assertEqual(
            set(filtered), {"transformer", "scheduler", "vae", "audio_vae"}
        )

    def test_mova_encoder_keeps_video_and_audio_vaes(self):
        extras = self._get_extra_modules(MOVAPipeline, RoleType.ENCODER, "i2v")
        filtered = filter_modules_for_role(
            MOVAPipeline._required_config_modules,
            RoleType.ENCODER,
            extra_allowed_modules=extras,
        )
        self.assertEqual(extras, {"video_vae", "audio_vae"})
        self.assertEqual(
            set(filtered),
            {"video_vae", "audio_vae", "text_encoder", "tokenizer", "scheduler"},
        )

    def test_mova_alias_uses_same_encoder_extras(self):
        extras = self._get_extra_modules(MOVAPipelineAlias, RoleType.ENCODER, "i2v")
        self.assertEqual(extras, {"video_vae", "audio_vae"})


class _GlobalStageArgsMixin:
    def setUp(self):
        super().setUp()
        self._prev_global_server_args = server_args_module._global_server_args
        set_global_server_args(SimpleNamespace(comfyui_mode=False))

    def tearDown(self):
        set_global_server_args(self._prev_global_server_args)
        super().tearDown()


class TestStageAffinityAndValidation(_GlobalStageArgsMixin, unittest.TestCase):
    def _make_hunyuan_pipeline(
        self, role: RoleType, *, paint_enable: bool
    ) -> Hunyuan3D2Pipeline:
        pipeline = object.__new__(Hunyuan3D2Pipeline)
        pipeline.server_args = SimpleNamespace(
            pipeline_config=Hunyuan3D2PipelineConfig(paint_enable=paint_enable)
        )
        pipeline._disagg_role = role
        pipeline.modules = {
            "hy3dshape_image_processor": object(),
            "hy3dshape_conditioner": object(),
            "hy3dshape_scheduler": object(),
            "hy3dshape_model": SimpleNamespace(
                parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))])
            ),
            "hy3dshape_vae": object(),
        }
        pipeline._stages = []
        pipeline._stage_name_mapping = {}
        return pipeline

    def test_helios_denoising_stage_is_denoiser_affine(self):
        stage = object.__new__(HeliosChunkedDenoisingStage)
        self.assertEqual(stage.role_affinity, RoleType.DENOISER)

    def test_mova_denoising_stage_is_denoiser_affine(self):
        stage = object.__new__(MOVADenoisingStage)
        self.assertEqual(stage.role_affinity, RoleType.DENOISER)

    def test_mova_decoding_stage_is_decoder_affine(self):
        stage = object.__new__(MOVADecodingStage)
        self.assertEqual(stage.role_affinity, RoleType.DECODER)

    def test_hunyuan3d_shape_only_disagg_accepts_non_monolithic_roles(self):
        pipeline = self._make_hunyuan_pipeline(RoleType.ENCODER, paint_enable=False)
        pipeline.validate_disagg_role(RoleType.ENCODER)
        pipeline.validate_disagg_role(RoleType.MONOLITHIC)

    def test_hunyuan3d_disagg_rejects_paint_pipeline(self):
        pipeline = self._make_hunyuan_pipeline(RoleType.ENCODER, paint_enable=True)
        with self.assertRaisesRegex(ValueError, "shape-only disaggregation"):
            pipeline.validate_disagg_role(RoleType.ENCODER)

    def test_hunyuan3d_shape_export_and_save_are_decoder_affine(self):
        export_stage = Hunyuan3DShapeExportStage(
            vae=object(),
            config=Hunyuan3D2PipelineConfig(paint_enable=False),
        )
        save_stage = Hunyuan3DShapeSaveStage(
            config=Hunyuan3D2PipelineConfig(paint_enable=False),
        )

        self.assertEqual(export_stage.role_affinity, RoleType.DECODER)
        self.assertEqual(save_stage.role_affinity, RoleType.DECODER)

    def test_hunyuan3d_stage_filtering_matches_shape_only_roles(self):
        expected = {
            RoleType.ENCODER: ["shape_before_denoising"],
            RoleType.DENOISER: ["shape_denoising"],
            RoleType.DECODER: ["shape_export", "shape_save"],
        }

        for role, stage_names in expected.items():
            pipeline = self._make_hunyuan_pipeline(role, paint_enable=False)
            pipeline.create_pipeline_stages(pipeline.server_args)
            self.assertEqual(list(pipeline._stage_name_mapping.keys()), stage_names)

    def test_hunyuan3d_shape_stage_no_longer_stores_model_dtype(self):
        pipeline = self._make_hunyuan_pipeline(RoleType.ENCODER, paint_enable=False)
        pipeline.create_pipeline_stages(pipeline.server_args)
        stage = pipeline._stage_name_mapping["shape_before_denoising"]
        self.assertIsInstance(stage, Hunyuan3DShapeBeforeDenoisingStage)
        self.assertFalse(hasattr(stage, "model_dtype"))


class TestHunyuan3DShapeStageRuntimeDtype(_GlobalStageArgsMixin, unittest.TestCase):
    def test_conditioner_parameter_dtype_wins_over_sample_dtype(self):
        conditioner = torch.nn.Linear(4, 4, bias=False).to(dtype=torch.float32)
        stage = Hunyuan3DShapeBeforeDenoisingStage(
            image_processor=object(),
            conditioner=conditioner,
            scheduler=SimpleNamespace(init_noise_sigma=1.0),
            config=Hunyuan3D2PipelineConfig(),
            latent_shape=(1, 2, 2),
            guidance_embed=False,
        )

        self.assertEqual(
            stage._resolve_runtime_dtype(torch.zeros(1, dtype=torch.float16)),
            torch.float32,
        )

    def test_runtime_dtype_falls_back_to_sample_tensor_without_module_dtype(self):
        stage = Hunyuan3DShapeBeforeDenoisingStage(
            image_processor=object(),
            conditioner=object(),
            scheduler=SimpleNamespace(init_noise_sigma=1.0),
            config=Hunyuan3D2PipelineConfig(),
            latent_shape=(1, 2, 2),
            guidance_embed=False,
        )

        self.assertEqual(
            stage._resolve_runtime_dtype(torch.zeros(1, dtype=torch.bfloat16)),
            torch.bfloat16,
        )


if __name__ == "__main__":
    unittest.main()
