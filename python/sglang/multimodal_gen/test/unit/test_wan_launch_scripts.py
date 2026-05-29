# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
from unittest.mock import patch
import unittest
from pathlib import Path
from types import SimpleNamespace


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
        cls.ddit_disagg_module = _load_script_module(
            "launch_ddit_disagg_wan_t2v.py"
        )

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

    def test_ddit_disagg_launcher_accepts_forced_switch(self):
        args = self.ddit_disagg_module.build_parser().parse_args(
            [
                "--model-path",
                "wan",
                "--ddit-schedule-policy",
                "forced_switch",
            ]
        )

        self.assertEqual(args.ddit_schedule_policy, "forced_switch")

    def test_ddit_disagg_launcher_defaults_max_slots_by_policy(self):
        forced_args = self.ddit_disagg_module.build_parser().parse_args(
            [
                "--model-path",
                "wan",
                "--ddit-schedule-policy",
                "forced_switch",
            ]
        )
        wsjf_args = self.ddit_disagg_module.build_parser().parse_args(
            [
                "--model-path",
                "wan",
                "--ddit-schedule-policy",
                "wsjf",
                "--ddit-window-size",
                "6",
            ]
        )

        self.assertEqual(
            self.ddit_disagg_module._resolve_disagg_max_slots(forced_args), 1
        )
        self.assertEqual(
            self.ddit_disagg_module._resolve_disagg_max_slots(wsjf_args), 6
        )

    def test_ddit_disagg_launcher_passes_buffer_and_warmup_args(self):
        args = self.ddit_disagg_module.build_parser().parse_args(
            [
                "--model-path",
                "wan",
                "--ddit-schedule-policy",
                "hungry_first",
                "--disagg-max-slots-per-instance",
                "3",
                "--disagg-transfer-pool-size",
                "536870912",
                "--disagg-transfer-redundancy",
                "1.5",
                "--disagg-warmup",
                "--disagg-warmup-resolutions",
                "1280x720,256x144",
                "--disagg-warmup-steps",
                "2",
            ]
        )
        kwargs = self.ddit_disagg_module._common_kwargs(args)

        self.assertEqual(kwargs["disagg_max_slots_per_instance"], 3)
        self.assertEqual(kwargs["disagg_transfer_pool_size"], 536870912)
        self.assertEqual(kwargs["disagg_transfer_redundancy"], 1.5)
        self.assertTrue(kwargs["warmup"])
        self.assertEqual(kwargs["warmup_resolutions"], ["1280x720", "256x144"])
        self.assertEqual(kwargs["warmup_steps"], 2)

    def test_ddit_disagg_launcher_passes_shortpath_sp_degree_map(self):
        args = self.ddit_disagg_module.build_parser().parse_args(
            [
                "--model-path",
                "wan",
                "--ddit-sp-degree-map",
                "shortpath",
            ]
        )
        kwargs = self.ddit_disagg_module._common_kwargs(args)

        self.assertEqual(kwargs["ddit_sp_degree_map"], "shortpath")

    def test_ddit_disagg_launcher_defaults_prebuild_by_policy(self):
        forced_args = self.ddit_disagg_module.build_parser().parse_args(
            [
                "--model-path",
                "wan",
                "--ddit-schedule-policy",
                "forced_switch",
            ]
        )
        hungry_args = self.ddit_disagg_module.build_parser().parse_args(
            [
                "--model-path",
                "wan",
                "--ddit-schedule-policy",
                "hungry_first",
            ]
        )

        self.assertFalse(
            self.ddit_disagg_module._common_kwargs(forced_args)[
                "ddit_prebuild_sp_groups"
            ]
        )
        self.assertTrue(
            self.ddit_disagg_module._common_kwargs(hungry_args)[
                "ddit_prebuild_sp_groups"
            ]
        )

    def test_ddit_disagg_launcher_accepts_explicit_prebuild_override(self):
        args = self.ddit_disagg_module.build_parser().parse_args(
            [
                "--model-path",
                "wan",
                "--ddit-schedule-policy",
                "forced_switch",
                "--ddit-prebuild-sp-groups",
                "true",
            ]
        )
        kwargs = self.ddit_disagg_module._common_kwargs(args)

        self.assertTrue(kwargs["ddit_prebuild_sp_groups"])

    def test_ddit_disagg_launcher_all_mode_enables_prebuild_and_non_offload(self):
        args = self.ddit_disagg_module.build_parser().parse_args(
            [
                "--model-path",
                "wan",
                "--ddit-schedule-policy",
                "forced_switch",
                "--ddit-dynamic-sp-prebuild-mode",
                "all",
            ]
        )
        kwargs = self.ddit_disagg_module._common_kwargs(args)

        self.assertTrue(kwargs["ddit_prebuild_sp_groups"])
        self.assertFalse(kwargs["dit_cpu_offload"])
        self.assertFalse(kwargs["dit_layerwise_offload"])
        self.assertFalse(kwargs["text_encoder_cpu_offload"])
        self.assertFalse(kwargs["image_encoder_cpu_offload"])
        self.assertFalse(kwargs["vae_cpu_offload"])
        self.assertFalse(kwargs["pin_cpu_memory"])

    def test_ddit_disagg_launcher_rewrites_role_result_base_after_head_port_adjustment(
        self,
    ):
        args = self.ddit_disagg_module.build_parser().parse_args(
            [
                "--model-path",
                "wan",
                "--host",
                "127.0.0.1",
                "--scheduler-port",
                "5555",
                "--num-gpus",
                "2",
            ]
        )

        def fake_from_kwargs(**kwargs):
            role = kwargs.get("disagg_role")
            if getattr(role, "value", role) == "server":
                kwargs = dict(kwargs)
                kwargs["scheduler_port"] = 7777
            return SimpleNamespace(**kwargs)

        with patch.object(
            self.ddit_disagg_module.ServerArgs,
            "from_kwargs",
            side_effect=fake_from_kwargs,
        ):
            encoder_args, ddit_worker_args, head_args = (
                self.ddit_disagg_module._resolve_launch_args(args, [0, 1])
            )

        self.assertEqual(head_args.scheduler_port, 7777)
        self.assertEqual(encoder_args.disagg_server_addr, "tcp://127.0.0.1:7777")
        self.assertEqual(ddit_worker_args.disagg_server_addr, "tcp://127.0.0.1:7777")
        self.assertEqual(
            head_args.encoder_urls,
            f"tcp://127.0.0.1:{encoder_args.scheduler_port}",
        )
        self.assertEqual(
            head_args.ddit_worker_urls,
            f"tcp://127.0.0.1:{ddit_worker_args.scheduler_port}",
        )

    def test_ddit_disagg_role_processes_are_not_daemonic(self):
        args = self.ddit_disagg_module.build_parser().parse_args(
            ["--model-path", "wan"]
        )
        fake_processes = []

        class FakeProcess:
            def __init__(self, *unused_args, **kwargs):
                del unused_args
                self.kwargs = kwargs
                fake_processes.append(self)

            def start(self):
                raise RuntimeError("stop before launching roles")

        fake_parser = argparse.Namespace(parse_args=lambda: args)
        with patch.object(
            self.ddit_disagg_module.mp,
            "get_context",
            return_value=argparse.Namespace(Process=FakeProcess),
        ), patch.object(
            self.ddit_disagg_module, "build_parser", return_value=fake_parser
        ), patch.object(self.ddit_disagg_module, "launch_disagg_server"):
            with self.assertRaisesRegex(RuntimeError, "stop before launching roles"):
                self.ddit_disagg_module.main()

        self.assertTrue(fake_processes)
        self.assertTrue(
            all(process.kwargs.get("daemon") is False for process in fake_processes)
        )


if __name__ == "__main__":
    unittest.main()
