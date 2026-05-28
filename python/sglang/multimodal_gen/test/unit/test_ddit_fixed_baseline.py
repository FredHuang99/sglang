# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.ddit.config import (
    build_execution_plan,
    resolve_vae_ranks,
    select_preferred_rank_tuple,
)
from sglang.multimodal_gen.runtime.ddit.scheduler import (
    DDiTRequestState,
    FixedBaselineScheduler,
    FixedBaselineSchedulerConfig,
    RequestPhase,
)


class TestDDiTFixedBaseline(unittest.TestCase):
    def _server_args(self, **overrides):
        values = {
            "ddit_schedule_policy": "fixed_baseline",
            "ddit_baseline_gpus": 4,
            "ddit_allowed_gpu_counts": "1,2,4,8",
            "ddit_local_ranks": None,
            "ddit_initial_gpus": 1,
            "ddit_initial_ranks": None,
            "ddit_switch_plan": "15:1->2;30:2->4",
            "ddit_vae_gpus": 1,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _batch(self, extra=None):
        return SimpleNamespace(extra=dict(extra or {}), request_id="req0")

    def test_fixed_plan_ignores_initial_gpus_and_switch_plan(self):
        plan = build_execution_plan(
            self._server_args(),
            self._batch(),
            world_size=8,
        )

        self.assertTrue(plan.enabled)
        self.assertEqual(plan.policy, "fixed_baseline")
        self.assertEqual(plan.initial_ranks, (0, 1, 2, 3))
        self.assertEqual(plan.switches, ())

    def test_fixed_plan_accepts_non_contiguous_explicit_ranks(self):
        plan = build_execution_plan(
            self._server_args(),
            self._batch({"ddit_baseline_ranks": [0, 2, 5, 7]}),
            world_size=8,
        )

        self.assertEqual(plan.initial_ranks, (0, 2, 5, 7))

    def test_fixed_vae_reuses_final_dit_ranks_and_ignores_overrides(self):
        ranks = resolve_vae_ranks(
            self._server_args(ddit_vae_gpus=1),
            self._batch({"ddit_vae_k": 1, "ddit_vae_ranks": "0"}),
            world_size=8,
            final_dit_ranks=(0, 2, 5, 7),
        )

        self.assertEqual(ranks, (0, 2, 5, 7))

    def test_rank_selection_prefers_contiguous_then_allows_gaps(self):
        self.assertEqual(select_preferred_rank_tuple((0, 2, 3, 4), 3), (2, 3, 4))
        self.assertEqual(select_preferred_rank_tuple((0, 2, 4, 6), 2), (0, 2))

    def test_fixed_baseline_scheduler_tracks_text_then_dit_vae_ranks(self):
        scheduler = FixedBaselineScheduler(
            FixedBaselineSchedulerConfig(
                local_ranks=tuple(range(8)),
                baseline_gpus=4,
                allowed_gpu_counts=(1, 2, 4, 8),
            )
        )
        for request_id in ("req_a", "req_b", "req_c"):
            scheduler.add_request(
                DDiTRequestState(
                    request_id=request_id,
                    resolution="720p",
                    total_steps=50,
                )
            )
            scheduler.mark_text_encoder_done(request_id)

        first_decisions = scheduler.schedule()
        self.assertEqual(first_decisions[0]["new_ranks"], (0, 1, 2, 3))
        self.assertEqual(first_decisions[1]["new_ranks"], (4, 5, 6, 7))
        self.assertEqual(scheduler.requests["req_c"].phase, RequestPhase.DIT_WAITING)

        scheduler.complete_request("req_a")
        second_decisions = scheduler.schedule()
        self.assertEqual(second_decisions[0]["request_id"], "req_c")
        self.assertEqual(second_decisions[0]["new_ranks"], (0, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
