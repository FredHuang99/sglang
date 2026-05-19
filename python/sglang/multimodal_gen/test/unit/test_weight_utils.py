# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.loader import weight_utils


class FakeSafetensorsStreamer:
    def __init__(self):
        self.calls = []
        self.current_file = None

    def __enter__(self):
        self.calls.append("enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        self.calls.append("exit")

    def stream_file(self, st_file):
        self.current_file = st_file
        self.calls.append(("stream_file", st_file))

    def stream_files(self, hf_weights_files):
        del hf_weights_files
        raise AssertionError("batch stream_files must not be used")

    def get_tensors(self):
        self.calls.append(("get_tensors", self.current_file))
        name = os.path.basename(str(self.current_file)) + ".weight"
        return iter([(name, torch.ones((1,), dtype=torch.float32))])


class FakeSafeOpen:
    def __init__(self, st_file):
        self.st_file = st_file

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def keys(self):
        return ["tensor.weight"]

    def get_tensor(self, name):
        del name
        return torch.ones((1,), dtype=torch.float32)


class TestSafetensorsWeightsIterator(unittest.TestCase):
    def test_runai_streams_one_file_at_a_time(self):
        fake_streamer = FakeSafetensorsStreamer()
        stages = []

        with (
            patch.object(weight_utils, "HAS_RUNAI_MODEL_STREAMER", True),
            patch.object(
                weight_utils,
                "SafetensorsStreamer",
                return_value=fake_streamer,
                create=True,
            ),
            patch.object(weight_utils, "_validate_safetensors_file", return_value=True),
            patch.dict(
                weight_utils.envs.environment_variables,
                {"SGLANG_USE_RUNAI_MODEL_STREAMER": lambda: True},
            ),
        ):
            tensors = list(
                weight_utils.safetensors_weights_iterator(
                    ["file0.safetensors", "file1.safetensors"],
                    stage_callback=lambda stage, detail=None: stages.append(
                        (stage, detail)
                    ),
                )
            )

        self.assertEqual(
            fake_streamer.calls,
            [
                "enter",
                ("stream_file", "file0.safetensors"),
                ("get_tensors", "file0.safetensors"),
                ("stream_file", "file1.safetensors"),
                ("get_tensors", "file1.safetensors"),
                "exit",
            ],
        )
        self.assertEqual(
            [name for name, _tensor in tensors],
            [
                "file0.safetensors.weight",
                "file1.safetensors.weight",
            ],
        )
        stage_names = [stage for stage, _detail in stages]
        self.assertIn("runai_stream_files_start", stage_names)
        self.assertIn("runai_stream_file_start", stage_names)
        self.assertIn("runai_stream_file_done", stage_names)
        self.assertIn("runai_get_tensors_start", stage_names)
        self.assertIn("runai_first_tensor", stage_names)
        self.assertIn("runai_stream_files_done", stage_names)

    def test_runai_env_gate_is_evaluated_at_call_time(self):
        with (
            patch.object(weight_utils, "HAS_RUNAI_MODEL_STREAMER", True),
            patch.dict(
                weight_utils.envs.environment_variables,
                {"SGLANG_USE_RUNAI_MODEL_STREAMER": lambda: False},
            ),
        ):
            self.assertFalse(weight_utils._resolve_use_runai_model_streamer(None))
            self.assertTrue(weight_utils._resolve_use_runai_model_streamer(True))
            self.assertFalse(weight_utils._resolve_use_runai_model_streamer(False))

    def test_explicit_false_disables_runai_streamer(self):
        safe_open_calls = []
        stages = []

        def fake_safe_open(st_file, framework, device):
            safe_open_calls.append((st_file, framework, device))
            return FakeSafeOpen(st_file)

        with (
            patch.object(weight_utils, "HAS_RUNAI_MODEL_STREAMER", True),
            patch.object(
                weight_utils,
                "SafetensorsStreamer",
                side_effect=AssertionError("RunAI streamer should stay disabled"),
                create=True,
            ),
            patch.object(weight_utils, "safe_open", side_effect=fake_safe_open),
            patch.object(weight_utils, "tqdm", lambda iterable, **kwargs: iterable),
            patch.dict(
                weight_utils.envs.environment_variables,
                {"SGLANG_USE_RUNAI_MODEL_STREAMER": lambda: True},
            ),
        ):
            tensors = list(
                weight_utils.safetensors_weights_iterator(
                    ["file0.safetensors"],
                    use_runai_model_streamer=False,
                    stage_callback=lambda stage, detail=None: stages.append(
                        (stage, detail)
                    ),
                )
            )

        self.assertEqual(len(tensors), 1)
        self.assertEqual(tensors[0][0], "tensor.weight")
        self.assertEqual(
            safe_open_calls,
            [("file0.safetensors", "pt", "cpu")],
        )
        self.assertEqual(stages, [])


if __name__ == "__main__":
    unittest.main()
