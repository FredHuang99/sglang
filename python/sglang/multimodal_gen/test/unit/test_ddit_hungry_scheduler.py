# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.ddit.config import (
    DDiTSwitchEvent,
    parse_sp_degree_map,
    resolve_ddit_sp_degrees,
    resolve_shortpath_sp_degrees,
    resolve_sp_degrees,
    resolve_vae_ranks,
)
from sglang.multimodal_gen.runtime.ddit.concurrent import CommandWaveBuilder, DDiTOp
from sglang.multimodal_gen.runtime.ddit.dynamic_sp import (
    DynamicSPBuildStats,
    DynamicSPGroupRegistry,
    LightweightDynamicSPCoordinator,
    _prebuild_rank_tuples,
    get_dynamic_sp_registry,
)
from sglang.multimodal_gen.runtime.ddit.profile import ProfileStore
from sglang.multimodal_gen.runtime.ddit.scheduler import (
    DDiTRequestState,
    DDiTSchedulerConfig,
    ForcedSwitchScheduler,
    ForcedSwitchSchedulerConfig,
    HungryFirstScheduler,
    NaiveGreedyScheduler,
    NaiveScheduler,
    ProfileSchedulerConfig,
    RequestPhase,
    WSJFScheduler,
    WSJFScaleUpScheduler,
    build_hungry_scheduler_config,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


class TestDDiTHungryScheduler(unittest.TestCase):
    def _profile_config(self, **kwargs):
        server_args = SimpleNamespace(
            ddit_profile_path=None,
            ddit_profile_model_id="wan2.1-t2v-1.3b",
            ddit_local_ranks=None,
            ddit_allowed_gpu_counts="1,2,4,8",
            model_id=None,
            model_path="wan2.1-t2v-1.3b",
        )
        profile = ProfileStore.load(server_args)
        return ProfileSchedulerConfig(
            local_ranks=kwargs.get("local_ranks", tuple(range(8))),
            allowed_gpu_counts=kwargs.get("allowed_gpu_counts", (1, 2, 4, 8)),
            profile=profile,
            window_size=kwargs.get("window_size", 8),
        )

    def test_wave_planner_allows_disjoint_request_steps(self):
        builder = CommandWaveBuilder(wave_id=7, world_size=8)

        self.assertTrue(
            builder.add(
                DDiTOp(
                    action="dit_step",
                    request_id="req_144",
                    ranks=(0,),
                    stage="dit",
                    step=3,
                )
            )
        )
        self.assertTrue(
            builder.add(
                DDiTOp(
                    action="dit_step",
                    request_id="req_720",
                    ranks=(1, 2, 3, 4),
                    stage="dit",
                    step=3,
                )
            )
        )
        self.assertFalse(
            builder.add(
                DDiTOp(
                    action="dit_step",
                    request_id="req_overlap",
                    ranks=(4, 5),
                    stage="dit",
                    step=3,
                )
            )
        )

        wave = builder.build()
        commands = wave.commands_by_rank(8)
        self.assertEqual(commands[0]["request_id"], "req_144")
        self.assertEqual(commands[1]["request_id"], "req_720")
        self.assertEqual(commands[5]["action"], "idle")

    def test_144p_completion_allows_720p_expand_from_4_to_8(self):
        scheduler = HungryFirstScheduler(
            DDiTSchedulerConfig(
                local_ranks=tuple(range(8)),
                opt_gpus_num={"144p": 1, "720p": 8},
                allowed_gpu_counts=(1, 2, 4, 8),
            )
        )
        scheduler.add_request(
            DDiTRequestState("req_144", resolution="144p", total_steps=50)
        )
        scheduler.add_request(
            DDiTRequestState("req_720", resolution="720p", total_steps=50)
        )

        first = scheduler.schedule()
        self.assertEqual(first[0]["new_ranks"], (0,))
        self.assertEqual(first[1]["new_ranks"], (1, 2, 3, 4))

        scheduler.complete_dit("req_144", vae_k=1)
        scheduler.complete_vae("req_144")
        scheduler.update_cur_step("req_720", 8)
        second = scheduler.schedule()

        self.assertEqual(second[0]["request_id"], "req_720")
        self.assertEqual(second[0]["old_ranks"], (1, 2, 3, 4))
        self.assertEqual(second[0]["new_ranks"], tuple(range(8)))
        self.assertEqual(second[0]["reason"], "hungry_first")

    def test_transition_to_vae_keeps_selected_final_dit_ranks(self):
        scheduler = HungryFirstScheduler(
            DDiTSchedulerConfig(local_ranks=tuple(range(8)))
        )
        scheduler.add_request(
            DDiTRequestState("req", resolution="720p", total_steps=50)
        )
        scheduler.schedule()
        scheduler.transition_to_vae("req", (0, 1))

        self.assertEqual(scheduler.requests["req"].phase, RequestPhase.VAE)
        self.assertEqual(scheduler.requests["req"].ranks, (0, 1))
        self.assertEqual(scheduler.gpu_owner[0], "req")
        self.assertEqual(scheduler.gpu_owner[1], "req")
        self.assertIsNone(scheduler.gpu_owner[2])

    def test_hungry_vae_default_selects_from_final_dit_ranks(self):
        server_args = SimpleNamespace(
            ddit_schedule_policy="hungry_first",
            ddit_allowed_gpu_counts="1,2,4,8",
            ddit_local_ranks=None,
            ddit_vae_gpus=2,
        )
        batch = SimpleNamespace(extra={}, request_id="req")
        ranks = resolve_vae_ranks(
            server_args,
            batch,
            world_size=8,
            final_dit_ranks=(2, 3, 4, 5),
        )

        self.assertEqual(ranks, (2, 3))

    def test_hungry_prepare_credit_requires_free_rank(self):
        scheduler = HungryFirstScheduler(
            DDiTSchedulerConfig(local_ranks=(0,), allowed_gpu_counts=(1,))
        )
        scheduler.add_request(
            DDiTRequestState("req_busy", resolution="144p", total_steps=50)
        )
        scheduler.schedule()

        self.assertEqual(scheduler.prepare_credit(), 0)

    def test_forced_switch_scheduler_follows_request_switch_plan(self):
        scheduler = ForcedSwitchScheduler(
            ForcedSwitchSchedulerConfig(
                local_ranks=tuple(range(4)),
                allowed_gpu_counts=(1, 2, 4),
            )
        )
        scheduler.add_request(
            DDiTRequestState(
                "req_forced",
                resolution="720p",
                total_steps=50,
                initial_ranks=(0,),
                switch_plan=(DDiTSwitchEvent(after_step=15, ranks=(0, 1)),),
            )
        )

        first = scheduler.schedule()
        self.assertEqual(first[0]["reason"], "forced_switch")
        self.assertEqual(first[0]["new_ranks"], (0,))

        scheduler.update_cur_step("req_forced", 15)
        second = scheduler.schedule()

        self.assertEqual(second[0]["old_ranks"], (0,))
        self.assertEqual(second[0]["new_ranks"], (0, 1))
        self.assertEqual(second[0]["reason"], "switch_plan")

    def test_dynamic_sp_registry_has_and_ensure_are_cache_aware(self):
        server_args = SimpleNamespace(
            ddit_sp_degree_map="1=1x1,2=2x1",
            ddit_profile_model_id="z-image",
            model_id="z-image",
            model_path="z-image",
        )
        registry = DynamicSPGroupRegistry(server_args)
        spec = registry.resolve_spec((1, 0))

        self.assertEqual(spec.ranks, (0, 1))
        self.assertEqual((spec.ulysses_degree, spec.ring_degree), (2, 1))
        self.assertFalse(registry.has((0, 1)))

        built_group = object()
        with patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.dist.is_available",
            return_value=True,
        ), patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.dist.is_initialized",
            return_value=True,
        ), patch.object(
            registry,
            "_build",
            return_value=(built_group, DynamicSPBuildStats(created_process_groups=1)),
        ) as build:
            first = registry.ensure((1, 0))
            second = registry.ensure((0, 1))

        self.assertTrue(first.created)
        self.assertIs(first.group, built_group)
        self.assertTrue(registry.has((0, 1)))
        self.assertFalse(second.created)
        self.assertIs(second.group, built_group)
        build.assert_called_once_with(spec)

    def test_dynamic_sp_singleton_build_uses_lightweight_local_bypass(self):
        server_args = SimpleNamespace(
            ddit_sp_degree_map="1=1x1",
            ddit_profile_model_id="z-image",
            model_id="z-image",
            model_path="z-image",
        )
        registry = DynamicSPGroupRegistry(server_args)

        with patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.dist.is_available",
            return_value=True,
        ), patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.dist.is_initialized",
            return_value=True,
        ), patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.dist.get_rank",
            return_value=0,
        ), patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.dist.new_group"
        ) as new_group, patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.get_world_group",
            return_value=SimpleNamespace(local_rank=0),
        ):
            result = registry.ensure((0,))

        self.assertTrue(result.created)
        self.assertIsInstance(result.group, LightweightDynamicSPCoordinator)
        self.assertEqual(result.group.world_size, 1)
        self.assertIsNone(result.group.device_group)
        self.assertEqual(result.stats.created_process_groups, 0)
        new_group.assert_not_called()

    def test_dynamic_sp_process_group_cache_deduplicates_repeated_ranks(self):
        server_args = SimpleNamespace(
            ddit_sp_degree_map="1=1x1,2=2x1",
            ddit_profile_model_id="z-image",
            model_id="z-image",
            model_path="z-image",
        )
        registry = DynamicSPGroupRegistry(server_args)
        calls = []

        def fake_new_group(**kwargs):
            calls.append(tuple(kwargs["ranks"]))
            return object()

        with patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.dist.is_available",
            return_value=True,
        ), patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.dist.is_initialized",
            return_value=True,
        ), patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.dist.get_rank",
            return_value=0,
        ), patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.dist.new_group",
            side_effect=fake_new_group,
        ), patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.get_world_group",
            return_value=SimpleNamespace(local_rank=0),
        ):
            result = registry.ensure((0, 1))

        self.assertTrue(result.created)
        self.assertEqual(calls, [(0, 1)])
        self.assertEqual(result.stats.created_process_groups, 1)
        self.assertEqual(result.stats.reused_process_groups, 1)

    def test_dynamic_sp_full_static_group_is_reused_without_ensure_wave(self):
        server_args = SimpleNamespace(
            ddit_sp_degree_map="1=1x1,2=2x1",
            ddit_profile_model_id="z-image",
            model_id="z-image",
            model_path="z-image",
        )
        registry = DynamicSPGroupRegistry(server_args)
        static_group = SimpleNamespace(
            ranks=[0, 1],
            ulysses_world_size=2,
            ring_world_size=1,
        )

        with patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.dist.is_available",
            return_value=True,
        ), patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.dist.is_initialized",
            return_value=True,
        ), patch(
            "sglang.multimodal_gen.runtime.distributed.parallel_state.get_sp_group",
            return_value=static_group,
        ), patch(
            "sglang.multimodal_gen.runtime.ddit.dynamic_sp.dist.new_group"
        ) as new_group:
            self.assertTrue(registry.has((0, 1)))
            result = registry.ensure((0, 1))

        self.assertFalse(result.created)
        self.assertIs(result.group, static_group)
        new_group.assert_not_called()

    def test_dynamic_sp_prebuild_uses_bounded_forced_switch_rank_tuples(self):
        server_args = SimpleNamespace(
            ddit_local_ranks=None,
            ddit_allowed_gpu_counts="1,2,4,8",
            ddit_initial_gpus=1,
            ddit_initial_ranks=None,
            ddit_switch_plan="15:1->2;30:2->4;45:4->8",
            ddit_vae_gpus=1,
            ddit_vae_ranks=None,
            ddit_baseline_gpus=None,
            ddit_baseline_ranks=None,
            ddit_schedule_policy="forced_switch",
            ddit_dynamic_sp_prebuild_mode="auto",
        )

        self.assertEqual(
            _prebuild_rank_tuples(server_args, world_size=8),
            ((0,), (0, 1), (0, 1, 2, 3), tuple(range(8))),
        )

    def test_dynamic_sp_forced_switch_auto_prebuild_skips_canonical_extras(self):
        server_args = SimpleNamespace(
            ddit_local_ranks="0,1,2,3,4,5,6,7",
            ddit_allowed_gpu_counts="1,2,4,8",
            ddit_initial_gpus=1,
            ddit_initial_ranks="2",
            ddit_switch_plan="10:2,3",
            ddit_vae_gpus=1,
            ddit_vae_ranks="6",
            ddit_baseline_gpus=None,
            ddit_baseline_ranks=None,
            ddit_schedule_policy="forced_switch",
            ddit_dynamic_sp_prebuild_mode="auto",
        )

        self.assertEqual(
            _prebuild_rank_tuples(server_args, world_size=8),
            ((2,), (6,), (2, 3)),
        )

    def test_dynamic_sp_prebuild_keeps_explicit_non_prefix_rank_tuples(self):
        server_args = SimpleNamespace(
            ddit_local_ranks="0,1,2,3,4,5,6,7",
            ddit_allowed_gpu_counts="1,2,4,8",
            ddit_initial_gpus=1,
            ddit_initial_ranks="2",
            ddit_switch_plan="10:2,3;20:1,3,5,7",
            ddit_vae_gpus=1,
            ddit_vae_ranks="6",
            ddit_baseline_gpus=None,
            ddit_baseline_ranks=None,
            ddit_schedule_policy="hungry_first",
            ddit_dynamic_sp_prebuild_mode="auto",
        )

        rank_tuples = _prebuild_rank_tuples(server_args, world_size=8)

        self.assertIn((2,), rank_tuples)
        self.assertIn((2, 3), rank_tuples)
        self.assertIn((1, 3, 5, 7), rank_tuples)
        self.assertIn((6,), rank_tuples)
        self.assertLess(len(rank_tuples), 107)

    def test_scheduler_skips_dynamic_sp_ensure_wave_on_cache_hit(self):
        server_args = SimpleNamespace(
            ddit_sp_degree_map="1=1x1",
            ddit_profile_model_id="z-image",
            model_id="z-image",
            model_path="z-image",
        )
        registry = get_dynamic_sp_registry(server_args)
        spec = registry.resolve_spec((0,))
        registry._cache[spec] = object()
        scheduler = object.__new__(Scheduler)
        scheduler.server_args = server_args

        pending_dynamic_sp = deque()
        ensured_dynamic_sp = set()
        ensuring_dynamic_sp = set()
        scheduler._ddit_queue_dynamic_sp_ensure(
            pending_dynamic_sp=pending_dynamic_sp,
            ensured_dynamic_sp=ensured_dynamic_sp,
            ensuring_dynamic_sp=ensuring_dynamic_sp,
            full_ranks=tuple(range(4)),
            target_ranks=(0,),
            request_id="req",
            reason="dit_init",
        )

        self.assertEqual(list(pending_dynamic_sp), [])
        self.assertEqual(ensured_dynamic_sp, {(0,)})
        self.assertEqual(ensuring_dynamic_sp, set())

    def test_scheduler_queues_full_rank_dynamic_sp_ensure_on_cache_miss(self):
        server_args = SimpleNamespace(
            ddit_sp_degree_map="1=1x1,2=2x1",
            ddit_profile_model_id="z-image",
            model_id="z-image",
            model_path="z-image",
        )
        scheduler = object.__new__(Scheduler)
        scheduler.server_args = server_args

        pending_dynamic_sp = deque()
        ensured_dynamic_sp = set()
        ensuring_dynamic_sp = set()
        scheduler._ddit_queue_dynamic_sp_ensure(
            pending_dynamic_sp=pending_dynamic_sp,
            ensured_dynamic_sp=ensured_dynamic_sp,
            ensuring_dynamic_sp=ensuring_dynamic_sp,
            full_ranks=tuple(range(4)),
            target_ranks=(1, 0),
            request_id="req",
            reason="dit_migrate",
        )

        self.assertEqual(len(pending_dynamic_sp), 1)
        op = pending_dynamic_sp[0]
        self.assertEqual(op.action, "ensure_dynamic_sp")
        self.assertEqual(op.ranks, tuple(range(4)))
        self.assertEqual(op.payload["target_ranks"], (0, 1))
        self.assertEqual(op.payload["log_reason"], "dit_migrate")
        self.assertEqual(ensured_dynamic_sp, set())
        self.assertEqual(ensuring_dynamic_sp, {(0, 1)})

    def test_disagg_ddit_register_prepared_returns_raw_outputs(self):
        scheduler = object.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(disagg_role="ddit_worker")
        scheduler.gpu_id = 0
        seen = {}

        def register(req):
            seen["save_output"] = req.save_output
            seen["return_file_paths_only"] = req.return_file_paths_only
            return {"request_id": req.request_id, "num_timesteps": 1}

        scheduler.worker = SimpleNamespace(register_hungry_prepared_request=register)

        result = scheduler._ddit_run_rank_command(
            {
                "action": "register_prepared",
                "scalar_fields": {
                    "request_id": "req",
                    "save_output": True,
                    "return_file_paths_only": True,
                },
                "tensors": {},
            }
        )

        self.assertEqual(result["request_id"], "req")
        self.assertFalse(seen["save_output"])
        self.assertFalse(seen["return_file_paths_only"])

    def test_ddit_result_forwarding_preserves_output_file_paths(self):
        scheduler = object.__new__(Scheduler)
        scheduler._pool_result_push = object()
        output_batch = OutputBatch(output_file_paths=["/tmp/out.png", None])

        with patch(
            "sglang.multimodal_gen.runtime.managers.scheduler.send_tensors"
        ) as send_tensors:
            scheduler._ddit_send_output_to_disagg_server("req", output_batch)

        send_tensors.assert_called_once()
        _socket, tensor_fields, scalar_fields = send_tensors.call_args.args
        self.assertEqual(tensor_fields, {})
        self.assertEqual(scalar_fields["request_id"], "req")
        self.assertEqual(scalar_fields["output_file_paths"], ["/tmp/out.png"])

    def test_forced_switch_waits_when_target_rank_is_busy(self):
        scheduler = ForcedSwitchScheduler(
            ForcedSwitchSchedulerConfig(
                local_ranks=tuple(range(4)),
                allowed_gpu_counts=(1, 2, 4),
            )
        )
        scheduler.add_request(
            DDiTRequestState(
                "req_a",
                resolution="720p",
                total_steps=50,
                initial_ranks=(0,),
                switch_plan=(DDiTSwitchEvent(after_step=15, ranks=(0, 1)),),
            )
        )
        scheduler.add_request(
            DDiTRequestState(
                "req_b",
                resolution="144p",
                total_steps=50,
                initial_ranks=(1,),
            )
        )
        scheduler.schedule()
        scheduler.update_cur_step("req_a", 15)

        self.assertEqual(scheduler.schedule(), [])

        scheduler.complete_request("req_b")
        decision = scheduler.schedule()
        self.assertEqual(decision[0]["request_id"], "req_a")
        self.assertEqual(decision[0]["new_ranks"], (0, 1))

    def test_profile_path_overrides_default_hungry_tables(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "opt_gpus_num": {"720p": 8},
                    "dit_step_times": {"720p": {"1": 10.0, "8": 2.0}},
                },
                f,
            )
            profile_path = f.name

        try:
            server_args = SimpleNamespace(
                ddit_local_ranks="0,1,2,3",
                ddit_allowed_gpu_counts="1,2,4,8",
                ddit_profile_path=profile_path,
            )
            config = build_hungry_scheduler_config(server_args, world_size=8)
        finally:
            os.unlink(profile_path)

        self.assertEqual(config.local_ranks, (0, 1, 2, 3))
        self.assertEqual(config.allowed_gpu_counts, (1, 2, 4))
        self.assertEqual(config.opt_gpus_num["720p"], 8)
        self.assertEqual(config.dit_step_times["720p"][8], 2.0)

    def test_naive_requires_head_request_opt_gpus(self):
        scheduler = NaiveScheduler(
            self._profile_config(
                local_ranks=(0, 1, 2, 3),
                allowed_gpu_counts=(1, 2, 4),
            )
        )
        scheduler.config.profile.opt_gpus_num["720p"] = 8
        scheduler.add_request(
            DDiTRequestState("req_720", resolution="720p", total_steps=50)
        )
        scheduler.add_request(
            DDiTRequestState("req_144", resolution="144p", total_steps=50)
        )

        self.assertEqual(scheduler.schedule(), [])
        self.assertEqual(list(scheduler.waiting), ["req_720", "req_144"])

    def test_prepare_credit_matches_policy_admission(self):
        naive = NaiveScheduler(
            self._profile_config(
                local_ranks=(0, 1, 2, 3),
                allowed_gpu_counts=(1, 2, 4),
            )
        )
        naive.config.profile.opt_gpus_num["720p"] = 8
        self.assertEqual(
            naive.prepare_credit(
                DDiTRequestState("req_720", resolution="720p", total_steps=50)
            ),
            0,
        )

        greedy = NaiveGreedyScheduler(
            self._profile_config(
                local_ranks=(0, 1, 2, 3),
                allowed_gpu_counts=(1, 2, 4),
            )
        )
        self.assertEqual(
            greedy.prepare_credit(
                DDiTRequestState("req_720", resolution="720p", total_steps=50)
            ),
            1,
        )

    def test_wsjf_prepare_credit_respects_window_and_free_ranks(self):
        scheduler = WSJFScheduler(self._profile_config(window_size=2))
        scheduler.add_request(
            DDiTRequestState("req_720", resolution="720p", total_steps=50)
        )
        scheduler.add_request(
            DDiTRequestState("req_144", resolution="144p", total_steps=50)
        )

        self.assertEqual(scheduler.prepare_credit(), 0)

        scheduler.schedule()
        self.assertGreaterEqual(scheduler.prepare_credit(), 1)

    def test_naive_greedy_downgrades_to_nearest_power_of_two(self):
        scheduler = NaiveGreedyScheduler(
            self._profile_config(local_ranks=(0, 1, 2, 3, 4))
        )
        scheduler.config.profile.opt_gpus_num["720p"] = 8
        scheduler.add_request(
            DDiTRequestState("req_720", resolution="720p", total_steps=50)
        )

        decisions = scheduler.schedule()

        self.assertEqual(decisions[0]["reason"], "naive_greedy")
        self.assertEqual(decisions[0]["new_ranks"], (0, 1, 2, 3))
        self.assertEqual(scheduler.requests["req_720"].ranks, (0, 1, 2, 3))

    def test_wsjf_picks_shortest_request_in_window(self):
        scheduler = WSJFScheduler(self._profile_config(window_size=3))
        scheduler.add_request(
            DDiTRequestState("req_720", resolution="720p", total_steps=50)
        )
        scheduler.add_request(
            DDiTRequestState("req_144", resolution="144p", total_steps=50)
        )
        scheduler.add_request(
            DDiTRequestState("req_360", resolution="360p", total_steps=50)
        )

        decisions = scheduler.schedule()

        self.assertEqual(decisions[0]["request_id"], "req_144")
        self.assertEqual(decisions[0]["reason"], "wsjf")

    def test_wsjf_scale_up_prioritizes_running_expansion(self):
        scheduler = WSJFScaleUpScheduler(self._profile_config())
        scheduler.add_request(
            DDiTRequestState("req_720", resolution="720p", total_steps=50)
        )
        first = scheduler.schedule()
        self.assertEqual(first[0]["new_ranks"], (0, 1, 2, 3))

        scheduler.add_request(
            DDiTRequestState("req_144", resolution="144p", total_steps=50)
        )
        scheduler.update_cur_step("req_720", 5)
        # Free two ranks and make opt=8 for this test so scale-up is meaningful.
        scheduler.config.profile.opt_gpus_num["720p"] = 8
        scheduler.config.profile.dit_step_times["720p"][8] = 3.0
        second = scheduler.schedule()

        self.assertEqual(second[0]["request_id"], "req_720")
        self.assertEqual(second[0]["reason"], "wsjf_scale_up")
        self.assertEqual(second[0]["old_ranks"], (0, 1, 2, 3))

    def test_profile_backed_policies_release_same_vae_ranks(self):
        for scheduler_cls in (
            NaiveScheduler,
            NaiveGreedyScheduler,
            WSJFScheduler,
            WSJFScaleUpScheduler,
        ):
            scheduler = scheduler_cls(self._profile_config())
            scheduler.add_request(
                DDiTRequestState("req", resolution="144p", total_steps=50)
            )
            scheduler.schedule()
            ranks = scheduler.requests["req"].ranks
            self.assertTrue(scheduler.vae_same_as_dit)
            scheduler.complete_request("req")
            for rank in ranks:
                self.assertIsNone(scheduler.gpu_owner[rank])

    def test_multi_model_profile_path_uses_requested_model(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "models": {
                        "z-image": {
                            "opt_gpus_num": {"720p": 2},
                            "dit_step_times": {"720p": {"2": 7.0}},
                        },
                        "wan2.1-t2v-1.3b": {
                            "opt_gpus_num": {"720p": 4},
                            "dit_step_times": {"720p": {"4": 3.0}},
                        },
                    }
                },
                f,
            )
            profile_path = f.name

        try:
            server_args = SimpleNamespace(
                ddit_profile_path=profile_path,
                ddit_profile_model_id="wan2.1-t2v-1.3b",
                model_id=None,
                model_path=None,
            )
            profile = ProfileStore.load(server_args)
        finally:
            os.unlink(profile_path)

        self.assertEqual(profile.model_id, "wan2.1-t2v-1.3b")
        self.assertEqual(profile.opt_gpus_num["720p"], 4)
        self.assertEqual(profile.per_step_time("unknown", 8), 1.0)

    def test_shortpath_sp_degree_table_for_wan(self):
        expected = {
            1: (1, 1),
            2: (2, 1),
            4: (4, 1),
            8: (2, 4),
        }
        for rank_count, degrees in expected.items():
            self.assertEqual(
                resolve_shortpath_sp_degrees("Wan2.1-T2V-1.3B", rank_count),
                degrees,
            )

    def test_shortpath_sp_degree_table_for_z_image(self):
        expected = {
            1: (1, 1),
            2: (2, 1),
            4: (2, 2),
            8: (2, 4),
        }
        for rank_count, degrees in expected.items():
            self.assertEqual(
                resolve_shortpath_sp_degrees("zimage", rank_count),
                degrees,
            )

    def test_shortpath_requires_supported_model_and_rank_count(self):
        with self.assertRaisesRegex(ValueError, "supports only"):
            resolve_shortpath_sp_degrees("unknown-model", 4)
        with self.assertRaisesRegex(ValueError, "rank counts"):
            resolve_shortpath_sp_degrees("z-image", 16)

    def test_ddit_sp_degree_shortpath_uses_server_args_model_id(self):
        server_args = SimpleNamespace(
            ddit_sp_degree_map="shortpath",
            ddit_profile_model_id="wan2.1-t2v-1.3b",
            model_id=None,
            model_path=None,
        )

        self.assertEqual(
            resolve_ddit_sp_degrees(
                8, server_args.ddit_sp_degree_map, server_args=server_args
            ),
            (2, 4),
        )

    def test_plain_sp_degree_map_behavior_is_unchanged(self):
        degree_map = "1=1x1,2=2x1,4=2x2"

        self.assertEqual(parse_sp_degree_map(degree_map)[4], (2, 2))
        self.assertEqual(resolve_sp_degrees(4, degree_map), (2, 2))
        self.assertEqual(resolve_sp_degrees(8, None), (8, 1))


if __name__ == "__main__":
    unittest.main()
