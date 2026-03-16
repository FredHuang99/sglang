# SPDX-License-Identifier: Apache-2.0
"""Unit tests for profiling matrix helpers."""

import unittest

from sglang.multimodal_gen.runtime.utils.profile_matrix import (
    build_ulysses_ring_pairs,
    validate_sp_topology,
)


class TestBuildUlyssesRingPairs(unittest.TestCase):
    """Test legal SP topology enumeration."""

    def test_build_pairs_for_eight(self):
        self.assertEqual(
            build_ulysses_ring_pairs(8),
            [(8, 1), (4, 2), (2, 4), (1, 8)],
        )

    def test_build_pairs_for_six_filtered_by_head_count(self):
        self.assertEqual(
            build_ulysses_ring_pairs(6, num_heads=40),
            [(2, 3), (1, 6)],
        )


class TestValidateSpTopology(unittest.TestCase):
    """Test SP topology validation logic."""

    def test_accepts_valid_topology(self):
        validate_sp_topology(
            num_gpus=8,
            sp_degree=8,
            ulysses_degree=4,
            ring_degree=2,
            num_heads=40,
        )

    def test_rejects_invalid_product(self):
        with self.assertRaisesRegex(ValueError, "must equal sp_degree"):
            validate_sp_topology(
                num_gpus=8,
                sp_degree=8,
                ulysses_degree=8,
                ring_degree=2,
            )

    def test_rejects_invalid_head_divisibility(self):
        with self.assertRaisesRegex(ValueError, "attention head count"):
            validate_sp_topology(
                num_gpus=6,
                sp_degree=6,
                ulysses_degree=3,
                ring_degree=2,
                num_heads=40,
            )


if __name__ == "__main__":
    unittest.main()
