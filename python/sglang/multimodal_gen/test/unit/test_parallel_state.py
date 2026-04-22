# SPDX-License-Identifier: Apache-2.0
"""Unit tests for distributed initialization device handling."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.distributed import parallel_state


def _fake_platform(
    *,
    cuda_alike: bool,
    device_name: str,
    mps: bool = False,
    musa: bool = False,
    npu: bool = False,
):
    return SimpleNamespace(
        is_cuda_alike=lambda: cuda_alike,
        is_mps=lambda: mps,
        is_musa=lambda: musa,
        is_npu=lambda: npu,
        device_name=device_name,
    )


class TestDistributedInitDeviceId(unittest.TestCase):
    def tearDown(self):
        parallel_state._WORLD = None

    def _run_init(self, *, platform, backend, device_id):
        fake_world = SimpleNamespace(world_size=1)
        with patch.object(
            parallel_state.torch.distributed,
            "is_initialized",
            return_value=False,
        ), patch.object(
            parallel_state.torch.distributed,
            "init_process_group",
        ) as init_process_group, patch.object(
            parallel_state.torch.distributed,
            "get_world_size",
            return_value=1,
        ), patch.object(
            parallel_state,
            "init_world_group",
            return_value=fake_world,
        ), patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform",
            platform,
        ):
            parallel_state._WORLD = None
            parallel_state.init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:12345",
                local_rank=0,
                backend=backend,
                device_id=device_id,
                timeout=12,
            )

        return init_process_group.call_args.kwargs

    def test_cpu_device_id_is_not_passed_to_process_group(self):
        kwargs = self._run_init(
            platform=_fake_platform(cuda_alike=False, device_name="CPU"),
            backend="gloo",
            device_id=torch.device("cpu"),
        )

        self.assertEqual(kwargs["backend"], "gloo")
        self.assertNotIn("device_id", kwargs)

    def test_nccl_backend_falls_back_to_gloo_without_device_id_on_cpu(self):
        kwargs = self._run_init(
            platform=_fake_platform(cuda_alike=False, device_name="CPU"),
            backend="nccl",
            device_id=torch.device("cpu"),
        )

        self.assertEqual(kwargs["backend"], "gloo")
        self.assertNotIn("device_id", kwargs)

    def test_indexed_cuda_device_id_is_passed_to_process_group(self):
        device_id = torch.device("cuda:1")
        kwargs = self._run_init(
            platform=_fake_platform(cuda_alike=True, device_name="cuda"),
            backend="nccl",
            device_id=device_id,
        )

        self.assertEqual(kwargs["backend"], "nccl")
        self.assertEqual(kwargs["device_id"], device_id)

    def test_musa_like_platform_keeps_omitting_device_id(self):
        kwargs = self._run_init(
            platform=_fake_platform(
                cuda_alike=True,
                device_name="musa",
                musa=True,
            ),
            backend="gloo",
            device_id=torch.device("cuda:0"),
        )

        self.assertEqual(kwargs["backend"], "gloo")
        self.assertNotIn("device_id", kwargs)


if __name__ == "__main__":
    unittest.main()
