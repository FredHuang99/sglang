# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.multimodal_gen.configs.models.encoders.qwen3 import Qwen3TextConfig
from sglang.multimodal_gen.runtime.models.encoders.qwen3 import _get_rope_parameters


class TestQwen3TextEncoderConfig(unittest.TestCase):
    def test_rope_parameters_are_created_from_legacy_fields(self):
        config = Qwen3TextConfig()
        config.arch_config.rope_theta = 1234.0
        config.arch_config.rope_scaling = {"rope_type": "default"}
        config.arch_config.extra_attrs.pop("rope_parameters", None)

        config.arch_config.__post_init__()
        rope_theta, rope_parameters = _get_rope_parameters(config)

        self.assertEqual(rope_theta, 1234.0)
        self.assertEqual(rope_parameters["rope_theta"], 1234.0)
        self.assertEqual(rope_parameters["rope_type"], "default")

    def test_existing_rope_parameters_are_preserved(self):
        config = Qwen3TextConfig()
        config.arch_config.rope_parameters = {
            "rope_theta": 777.0,
            "rope_type": "yarn",
        }

        config.arch_config.__post_init__()
        rope_theta, rope_parameters = _get_rope_parameters(config)

        self.assertEqual(rope_theta, 777.0)
        self.assertEqual(rope_parameters["rope_type"], "yarn")


if __name__ == "__main__":
    unittest.main()
