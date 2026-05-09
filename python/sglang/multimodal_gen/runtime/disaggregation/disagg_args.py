# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible disaggregated diffusion argument helpers.

The canonical disagg CLI fields and helper methods live in
``sglang.multimodal_gen.runtime.server_args``.  This module is kept as a thin
adapter while the disaggregation refactor is split across reviewable PRs, so
older imports from the v1 orchestrator path continue to work.
"""

from __future__ import annotations

import argparse

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.server_args import DisaggServerArgsMixin

DISAGG_RESULT_PORT_OFFSETS = DisaggServerArgsMixin.DISAGG_RESULT_PORT_OFFSETS
DisaggArgsMixin = DisaggServerArgsMixin


def add_disagg_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register disaggregated-diffusion CLI args through the canonical mixin."""

    DisaggServerArgsMixin.add_disagg_cli_args(parser)


def convert_disagg_role_string(kwargs: dict) -> None:
    """Convert ``disagg_role`` from string to ``RoleType`` enum in-place."""

    if "disagg_role" in kwargs and isinstance(kwargs["disagg_role"], str):
        kwargs["disagg_role"] = RoleType.from_string(kwargs["disagg_role"])
