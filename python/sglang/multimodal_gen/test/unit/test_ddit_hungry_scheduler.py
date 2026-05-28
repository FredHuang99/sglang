# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.ddit.config import resolve_vae_ranks
from sglang.multimodal_gen.runtime.ddit.concurrent import CommandWaveBuilder, DDiTOp
from sglang.multimodal_gen.runtime.ddit.scheduler import (
    DDiTRequestState,
    DDiTSchedulerConfig,
    HungryFirstScheduler,
    RequestPhase,
    build_hungry_scheduler_config,
)


class TestDDiTHungryScheduler(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
