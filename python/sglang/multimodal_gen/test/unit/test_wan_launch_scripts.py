# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import unittest
from pathlib import Path


def _load_script_module(script_name: str):
    root_dir = Path(__file__).resolve().parents[5]
    script_path = root_dir / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestWanLaunchScripts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.monolithic_module = _load_script_module(
            "launch_wan22_ti2v_monolithic_4gpu.py"
        )
        cls.disagg_module = _load_script_module("launch_pool_wan22_ti2v_4gpu.py")

    def test_launchers_disable_server_warmup_by_default(self):
        mono_args = self.monolithic_module.build_parser().parse_args([])
        disagg_args = self.disagg_module.build_parser().parse_args([])
        self.assertFalse(mono_args.warmup)
        self.assertFalse(disagg_args.warmup)

    def test_monolithic_gpu_ids_override_base_gpu_range(self):
        args = self.monolithic_module.build_parser().parse_args(
            ["--gpu-ids", "0", "1", "6", "7"]
        )
        self.assertEqual(self.monolithic_module._resolve_gpu_group(args), [0, 1, 6, 7])

    def test_gpu_ids_accept_comma_separated_values(self):
        args = self.disagg_module.build_parser().parse_args(
            ["--denoiser-gpu-ids", "0,1,6,7"]
        )
        self.assertEqual(
            self.disagg_module._resolve_role_gpu_ids(args, "denoiser"),
            [0, 1, 6, 7],
        )

    def test_cpu_encoder_rejects_explicit_gpu_ids(self):
        args = self.disagg_module.build_parser().parse_args(
            ["--encoder-device", "cpu", "--encoder-gpu-ids", "0", "1", "6", "7"]
        )
        with self.assertRaisesRegex(ValueError, "encoder-gpu-ids"):
            self.disagg_module._resolve_encoder_gpu_ids(args)

    def test_cpu_encoder_forces_tp_one(self):
        args = argparse.Namespace(
            encoder_device="cpu",
            encoder_tp=None,
            encoder_num_gpus=4,
        )
        self.assertEqual(self.disagg_module._resolve_encoder_tp(args), 1)

    def test_cpu_encoder_rejects_tp_greater_than_one(self):
        args = argparse.Namespace(
            encoder_device="cpu",
            encoder_tp=4,
            encoder_num_gpus=4,
        )
        with self.assertRaisesRegex(ValueError, "encoder_tp must be 1"):
            self.disagg_module._resolve_encoder_tp(args)

    def test_split_two_hosts_defaults_to_mixed_backends(self):
        args = argparse.Namespace(
            deployment_layout="split_two_hosts",
            disagg_transfer_backend=None,
            encoder_transfer_backend=None,
            denoiser_transfer_backend=None,
            decoder_transfer_backend=None,
        )
        self.assertEqual(
            self.disagg_module._resolve_role_transfer_backend(args, "encoder"),
            "mock",
        )
        self.assertEqual(
            self.disagg_module._resolve_role_transfer_backend(args, "denoiser"),
            "auto",
        )
        self.assertEqual(
            self.disagg_module._resolve_role_transfer_backend(args, "decoder"),
            "auto",
        )

    def test_single_host_defaults_to_mock_backend(self):
        args = argparse.Namespace(
            deployment_layout="single_host",
            disagg_transfer_backend=None,
            encoder_transfer_backend=None,
            denoiser_transfer_backend=None,
            decoder_transfer_backend=None,
        )
        self.assertEqual(
            self.disagg_module._resolve_role_transfer_backend(args, "denoiser"),
            "mock",
        )


if __name__ == "__main__":
    unittest.main()
