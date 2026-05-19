# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.loader.weight_broadcast import (
    broadcast_module_tensors,
    iter_module_tensors,
    materialize_empty_model_state_dict,
    resolve_rank0_broadcast_decision,
    validate_broadcast_metadata,
)
from sglang.multimodal_gen.runtime.utils.weight_load_profiler import (
    WEIGHT_LOAD_BROADCAST_BYTES,
    WEIGHT_LOAD_BROADCAST_ERROR,
    WEIGHT_LOAD_BROADCAST_TENSOR_COUNT,
    WEIGHT_LOAD_MODE_EFFECTIVE,
    WEIGHT_LOAD_MODE_REQUESTED,
    WEIGHT_LOAD_NCCL_BROADCAST_MS,
    DiffusionWeightLoadProfiler,
)


class FakeSPGroup:
    def __init__(self, *, rank_in_group=0, rank0_metadata=None):
        self.rank_in_group = rank_in_group
        self.world_size = 2
        self.device = torch.device("cpu")
        self.rank0_metadata = rank0_metadata
        self.broadcast_calls = 0

    def broadcast_object(self, obj=None, src=0):
        del src
        if self.rank_in_group == 0:
            self.rank0_metadata = obj
            return obj
        return self.rank0_metadata

    def all_reduce(self, tensor, op=None):
        del op
        return tensor

    def broadcast(self, tensor, src=0):
        del src
        self.broadcast_calls += 1
        return tensor


class TestDiffusionWeightBroadcast(unittest.TestCase):
    def _profile(self):
        return DiffusionWeightLoadProfiler(
            component="transformer",
            server_args=SimpleNamespace(
                profile_enabled=True,
                num_gpus=2,
                disagg_role_device="cuda",
            ),
            enabled=True,
        )

    def test_precondition_falls_back_when_component_not_enabled(self):
        decision = resolve_rank0_broadcast_decision(
            load_mode="rank0-broadcast",
            broadcast_components=[],
            component_name="transformer",
            tp_size=1,
            fsdp_inference=False,
        )

        self.assertFalse(decision.enabled)
        self.assertEqual(decision.requested_mode, "rank0-broadcast")
        self.assertEqual(decision.effective_mode, "default")
        self.assertIn("component_not_enabled", decision.reason)

    def test_precondition_falls_back_for_tp(self):
        decision = resolve_rank0_broadcast_decision(
            load_mode="rank0-broadcast",
            broadcast_components=["transformer"],
            component_name="transformer",
            tp_size=2,
            fsdp_inference=False,
        )

        self.assertFalse(decision.enabled)
        self.assertEqual(decision.effective_mode, "default")
        self.assertIn("tp_size_not_supported", decision.reason)

    def test_empty_materialization_replaces_meta_tensors(self):
        with torch.device("meta"):
            module = torch.nn.Sequential(
                torch.nn.Linear(2, 3),
                torch.nn.LayerNorm(3),
            )

        materialize_empty_model_state_dict(
            module,
            device=torch.device("cpu"),
            strict=True,
        )

        for name, tensor in module.state_dict().items():
            self.assertFalse(tensor.is_meta, name)
            self.assertEqual(tensor.device.type, "cpu")

    def test_metadata_detects_mismatch_before_tensor_broadcast(self):
        entries = [
            ("a", torch.empty((2, 2), dtype=torch.float32), "parameter"),
        ]
        rank0_metadata = [
            ("a", (3, 2), "torch.float32", "parameter"),
        ]
        group = FakeSPGroup(rank_in_group=1, rank0_metadata=rank0_metadata)
        profile = self._profile()

        with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
            validate_broadcast_metadata(
                entries,
                group,
                weight_load_profile=profile,
            )

        record = profile.as_dict()
        self.assertIn("metadata mismatch", record[WEIGHT_LOAD_BROADCAST_ERROR])

    def test_broadcast_records_tensor_count_and_bytes(self):
        module = torch.nn.Linear(2, 2)
        group = FakeSPGroup(rank_in_group=0)
        profile = self._profile()

        broadcast_module_tensors(module, group, weight_load_profile=profile)

        record = profile.as_dict()
        self.assertEqual(record[WEIGHT_LOAD_MODE_REQUESTED], "default")
        self.assertEqual(record[WEIGHT_LOAD_MODE_EFFECTIVE], "default")
        self.assertEqual(record[WEIGHT_LOAD_BROADCAST_TENSOR_COUNT], 2)
        self.assertEqual(
            record[WEIGHT_LOAD_BROADCAST_BYTES],
            sum(t.numel() * t.element_size() for t in module.state_dict().values()),
        )
        self.assertGreaterEqual(record[WEIGHT_LOAD_NCCL_BROADCAST_MS], 0.0)
        self.assertEqual(group.broadcast_calls, 3)

    def test_iter_module_tensors_is_stably_sorted_by_name(self):
        module = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.LayerNorm(2),
        )

        entries = iter_module_tensors(module)
        names = [name for name, _tensor, _kind in entries]

        self.assertEqual(names, sorted(names))


if __name__ == "__main__":
    unittest.main()
