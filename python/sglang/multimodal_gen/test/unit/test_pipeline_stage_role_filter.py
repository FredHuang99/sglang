# SPDX-License-Identifier: Apache-2.0
"""Unit tests for disaggregated pipeline stage construction filtering."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)


class _FakePipeline(ComposedPipelineBase):
    pipeline_name = "FakePipeline"
    _required_config_modules = []

    def initialize_pipeline(self, server_args):
        pass

    def create_pipeline_stages(self, server_args) -> None:
        pass


def _make_pipeline(role: RoleType) -> _FakePipeline:
    pipeline = object.__new__(_FakePipeline)
    pipeline.modules = {}
    pipeline._stages = []
    pipeline._stage_name_mapping = {}
    pipeline._disagg_role = role
    return pipeline


class TestPipelineStageRoleFilter(unittest.TestCase):
    def test_stage_factory_skips_without_constructing_for_other_role(self):
        pipeline = _make_pipeline(RoleType.ENCODER)

        def should_not_construct():
            raise AssertionError("stage factory should have been skipped")

        pipeline.add_stage_factory(
            RoleType.DENOISER,
            should_not_construct,
            "denoising_stage",
        )

        self.assertEqual(pipeline.stages, [])

    def test_stage_factory_constructs_for_matching_role(self):
        pipeline = _make_pipeline(RoleType.DENOISER)
        stage = SimpleNamespace(role_affinity=RoleType.DENOISER)
        events = []

        def create_stage():
            events.append("called")
            return stage

        pipeline.add_stage_factory(
            RoleType.DENOISER,
            create_stage,
            "denoising_stage",
        )

        self.assertEqual(events, ["called"])
        self.assertIs(pipeline.get_stage("denoising_stage"), stage)

    def test_encoder_role_does_not_construct_standard_denoising_stage(self):
        pipeline = _make_pipeline(RoleType.ENCODER)

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base.DenoisingStage",
            side_effect=AssertionError("DenoisingStage should not be constructed"),
        ):
            pipeline.add_standard_denoising_stage()

        self.assertEqual(pipeline.stages, [])

    def test_encoder_role_does_not_construct_standard_decoding_stage(self):
        pipeline = _make_pipeline(RoleType.ENCODER)

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base.DecodingStage",
            side_effect=AssertionError("DecodingStage should not be constructed"),
        ):
            pipeline.add_standard_decoding_stage()

        self.assertEqual(pipeline.stages, [])


if __name__ == "__main__":
    unittest.main()
