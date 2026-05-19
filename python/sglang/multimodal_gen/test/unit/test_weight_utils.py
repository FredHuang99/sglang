# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.loader import weight_utils


class FakeSafetensorsStreamer:
    def __init__(self):
        self.calls = []
        self.streamed_files = []

    def __enter__(self):
        self.calls.append("enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        self.calls.append("exit")

    def stream_file(self, st_file):
        self.calls.append(("stream_file", st_file))

    def stream_files(self, hf_weights_files):
        self.streamed_files = list(hf_weights_files)
        self.calls.append(("stream_files", tuple(hf_weights_files)))

    def get_tensors(self):
        self.calls.append("get_tensors")
        return iter(
            [
                (
                    os.path.basename(str(st_file)) + ".weight",
                    torch.ones((1,), dtype=torch.float32),
                )
                for st_file in self.streamed_files
            ]
        )


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
    def test_runai_uses_batch_stream_files_for_default_path(self):
        fake_streamer = FakeSafetensorsStreamer()

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
                )
            )

        self.assertEqual(
            fake_streamer.calls,
            [
                "enter",
                (
                    "stream_files",
                    ("file0.safetensors", "file1.safetensors"),
                ),
                "get_tensors",
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
            patch.object(weight_utils, "_validate_safetensors_file", return_value=True),
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
                )
            )

        self.assertEqual(len(tensors), 1)
        self.assertEqual(tensors[0][0], "tensor.weight")
        self.assertEqual(
            safe_open_calls,
            [("file0.safetensors", "pt", "cpu")],
        )


if __name__ == "__main__":
    unittest.main()
