# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_client_module():
    root_dir = Path(__file__).resolve().parents[5]
    script_path = root_dir / "examples" / "multimodal_gen" / "ddit_mixed_workload_client.py"
    spec = importlib.util.spec_from_file_location("ddit_mixed_workload_client", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestDDiTMixedWorkloadClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_client_module()
        cls.root_dir = Path(__file__).resolve().parents[5]

    def test_detect_project_root_from_script_location(self):
        self.assertEqual(self.module.detect_project_root(), self.root_dir)

    def test_default_image_path_uses_project_root(self):
        expected = self.root_dir / "examples" / "assets" / "example_image.png"
        self.assertEqual(Path(self.module.default_project_image_path()), expected)

    def test_relative_image_path_resolves_against_project_root(self):
        resolved = self.module.resolve_project_image_path(
            "examples/assets/example_image.png",
            project_root=str(self.root_dir),
        )
        self.assertEqual(
            Path(resolved),
            self.root_dir / "examples" / "assets" / "example_image.png",
        )

    def test_empty_image_path_disables_image_payload(self):
        self.assertIsNone(
            self.module.resolve_project_image_path("", project_root=str(self.root_dir))
        )


if __name__ == "__main__":
    unittest.main()
