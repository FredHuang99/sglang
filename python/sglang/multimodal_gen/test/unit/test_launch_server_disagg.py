# SPDX-License-Identifier: Apache-2.0
"""Unit tests for disaggregated launch-time calibration helpers."""

import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.runtime.launch_server import (
    _build_disagg_calibration_reqs,
    _run_disagg_startup_calibration,
)


class TestDisaggStartupCalibrationHelpers(unittest.TestCase):
    def test_build_calibration_reqs_for_image_task(self):
        server_args = SimpleNamespace(
            warmup=True,
            warmup_resolutions=["640x480"],
            pipeline_config=SimpleNamespace(task_type=ModelTaskType.I2I),
            warmup_steps=3,
        )

        with patch(
            "sglang.multimodal_gen.runtime.launch_server.asyncio.run",
            return_value="outputs/uploads/warmup_image.jpg",
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.save_image_to_path",
            new=MagicMock(return_value="ignored"),
        ):
            warmup_reqs = _build_disagg_calibration_reqs(server_args)

        self.assertEqual(len(warmup_reqs), 1)
        req = warmup_reqs[0]
        self.assertTrue(req.is_warmup)
        self.assertEqual(req.width, 640)
        self.assertEqual(req.height, 480)
        self.assertEqual(req.image_path, ["outputs/uploads/warmup_image.jpg"])
        self.assertEqual(req.negative_prompt, "")
        self.assertEqual(req.num_inference_steps, 3)

    def test_run_startup_calibration_sends_all_requests(self):
        server_args = SimpleNamespace(
            warmup=True,
            disagg_timeout=12,
        )
        fake_socket = MagicMock()
        fake_context = MagicMock()
        fake_context.socket.return_value = fake_socket
        fake_socket.recv_multipart.side_effect = [
            [b"id", pickle.dumps(SimpleNamespace(error=None))],
            [b"id", pickle.dumps(SimpleNamespace(error=None))],
        ]

        with patch(
            "sglang.multimodal_gen.runtime.launch_server._build_disagg_calibration_reqs",
            return_value=[SimpleNamespace(name="r1"), SimpleNamespace(name="r2")],
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.zmq.Context",
            return_value=fake_context,
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.time.sleep"
        ):
            _run_disagg_startup_calibration("tcp://127.0.0.1:9999", server_args)

        self.assertEqual(fake_socket.send.call_count, 2)
        fake_socket.connect.assert_called_once_with("tcp://127.0.0.1:9999")
        fake_socket.close.assert_called_once_with(linger=0)
        fake_context.destroy.assert_called_once_with(linger=0)

    def test_run_startup_calibration_fails_fast_on_error_reply(self):
        server_args = SimpleNamespace(
            warmup=True,
            disagg_timeout=5,
        )
        fake_socket = MagicMock()
        fake_context = MagicMock()
        fake_context.socket.return_value = fake_socket
        fake_socket.recv_multipart.return_value = [
            b"id",
            pickle.dumps(SimpleNamespace(error="calibration failed")),
        ]

        with patch(
            "sglang.multimodal_gen.runtime.launch_server._build_disagg_calibration_reqs",
            return_value=[SimpleNamespace(name="r1")],
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.zmq.Context",
            return_value=fake_context,
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.time.sleep"
        ), self.assertRaisesRegex(RuntimeError, "Disagg startup calibration failed"):
            _run_disagg_startup_calibration("tcp://127.0.0.1:9999", server_args)


if __name__ == "__main__":
    unittest.main()
