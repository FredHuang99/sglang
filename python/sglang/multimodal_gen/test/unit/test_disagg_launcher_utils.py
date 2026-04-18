# SPDX-License-Identifier: Apache-2.0

import json
import unittest

from sglang.multimodal_gen.runtime.utils.disagg_launcher_utils import (
    build_auto_ib_device_map,
    normalize_disagg_host,
    resolve_disagg_ib_device,
)


class TestDisaggLauncherUtils(unittest.TestCase):
    def test_normalize_disagg_host_accepts_hostname_alias(self):
        self.assertEqual(normalize_disagg_host("ac-h200-gpu03"), "10.3.4.3")
        self.assertEqual(normalize_disagg_host("10.3.4.2"), "10.3.4.2")

    def test_build_auto_ib_device_map_for_known_host(self):
        ib_map = build_auto_ib_device_map("10.3.4.3")
        self.assertEqual(ib_map[4], "mlx5_6")
        self.assertEqual(ib_map[7], "mlx5_9")

    def test_resolve_disagg_ib_device_auto_known_host_returns_json_mapping(self):
        resolved = resolve_disagg_ib_device("auto", host="10.3.4.2")
        self.assertIsNotNone(resolved)
        parsed = json.loads(resolved)
        self.assertEqual(parsed["4"], "mlx5_6")
        self.assertEqual(parsed["7"], "mlx5_9")

    def test_resolve_disagg_ib_device_auto_unknown_host_returns_none(self):
        self.assertIsNone(resolve_disagg_ib_device("auto", host="10.9.9.9"))

    def test_resolve_disagg_ib_device_preserves_explicit_value(self):
        self.assertEqual(
            resolve_disagg_ib_device("mlx5_0", host="10.3.4.3"),
            "mlx5_0",
        )


if __name__ == "__main__":
    unittest.main()
