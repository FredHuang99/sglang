# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import tempfile
import threading
import time
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

    def test_auto_input_reference_skips_text_only_models(self):
        for model_id in ("Z-Image", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"):
            resolved = self.module.resolve_input_reference_path(
                image_path=None,
                mode="auto",
                model_id=model_id,
                project_root=str(self.root_dir),
            )
            self.assertIsNone(resolved)

        workload = self.module.build_workload(
            num_requests=1,
            resolutions=["720p"],
            ratios=[1.0],
            seed=1,
            prompt="test",
            size_map={"720p": "1280x720"},
            image_path=None,
            include_input_reference=False,
        )

        self.assertNotIn("input_reference", workload[0].payload)
        self.assertNotIn("image_path", workload[0].payload)

    def test_auto_input_reference_uses_default_image_for_ti2v_models(self):
        resolved = self.module.resolve_input_reference_path(
            image_path=None,
            mode="auto",
            model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            project_root=str(self.root_dir),
        )
        self.assertEqual(
            Path(resolved),
            self.root_dir / "examples" / "assets" / "example_image.png",
        )

        workload = self.module.build_workload(
            num_requests=1,
            resolutions=["720p"],
            ratios=[1.0],
            seed=1,
            prompt="test",
            size_map={"720p": "1280x720"},
            image_path=resolved,
            include_input_reference=True,
        )

        self.assertEqual(workload[0].payload["input_reference"], resolved)
        self.assertNotIn("image_path", workload[0].payload)

    def test_forced_image_mode_rejects_empty_image_path(self):
        with self.assertRaisesRegex(ValueError, "requires a non-empty --image-path"):
            self.module.resolve_input_reference_path(
                image_path="",
                mode="image",
                model_id="z-image",
                project_root=str(self.root_dir),
            )

    def test_constant_ddit_vae_k_is_added_to_each_payload(self):
        workload = self.module.build_workload(
            num_requests=2,
            resolutions=["144p", "720p"],
            ratios=[0.5, 0.5],
            seed=1,
            prompt="test",
            size_map={"144p": "256x144", "720p": "1280x720"},
            ddit_vae_k_resolver=self.module.build_vae_k_resolver("1"),
        )

        self.assertEqual({item.payload["ddit_vae_k"] for item in workload}, {1})

    def test_ddit_vae_k_inline_map_uses_resolution(self):
        resolver = self.module.build_vae_k_resolver('{"144p":1,"720p":8}')
        workload = self.module.build_workload(
            num_requests=2,
            resolutions=["144p", "720p"],
            ratios=[0.5, 0.5],
            seed=1,
            prompt="test",
            size_map={"144p": "256x144", "720p": "1280x720"},
            ddit_vae_k_resolver=resolver,
        )
        by_resolution = {
            item.resolution: item.payload["ddit_vae_k"] for item in workload
        }

        self.assertEqual(by_resolution["144p"], 1)
        self.assertEqual(by_resolution["720p"], 8)

    def test_ddit_vae_k_profile_uses_opt_vae_k(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "models": {
                        "wan2.1-t2v-1.3b": {
                            "opt_vae_k": {"144p": 1, "360p": 4, "720p": 8}
                        }
                    }
                },
                f,
            )
            profile_path = f.name

        try:
            resolver = self.module.build_vae_k_resolver(
                "profile",
                profile_path=profile_path,
                profile_model_id="Wan2.1-T2V-1.3B",
            )
            self.assertEqual(resolver("144p"), 1)
            self.assertEqual(resolver("360p"), 4)
            self.assertEqual(resolver("720p"), 8)
        finally:
            Path(profile_path).unlink()

    def test_ddit_vae_k_profile_falls_back_to_one_without_opt_vae_k(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"models": {"z-image": {"opt_gpus_num": {"720p": 4}}}}, f)
            profile_path = f.name

        try:
            resolver = self.module.build_vae_k_resolver(
                "profile",
                profile_path=profile_path,
                profile_model_id="z-image",
            )
            self.assertEqual(resolver("720p"), 1)
            self.assertEqual(resolver("2k"), 1)
        finally:
            Path(profile_path).unlink()

    def test_ddit_vae_k_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "DDiT VAE k"):
            self.module.build_vae_k_resolver("3")
        with self.assertRaisesRegex(ValueError, "DDiT VAE k"):
            self.module.build_vae_k_resolver('{"720p":3}')("720p")

    def test_burst_send_is_concurrent_and_preserves_output_order(self):
        workload = [
            self.module.WorkloadRequest(
                f"req_{idx}", "144p", {"request_id": f"req_{idx}"}
            )
            for idx in range(3)
        ]

        class FakeResponse:
            status_code = 200
            text = "{}"

            def __init__(self, request_id):
                self.request_id = request_id

            def json(self):
                return {"request_id": self.request_id}

        def fake_post(_endpoint, *, json, timeout):
            del timeout
            time.sleep(0.1)
            return FakeResponse(json["request_id"])

        start = time.perf_counter()
        responses = self.module.send_workload(
            server_url="http://127.0.0.1:30000",
            workload=workload,
            rate=None,
            timeout=10,
            max_inflight=3,
            post_fn=fake_post,
        )
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.25)
        self.assertEqual(
            [record["client_request_id"] for record in responses],
            ["req_0", "req_1", "req_2"],
        )

    def test_rate_controls_submit_interval_not_response_completion(self):
        workload = [
            self.module.WorkloadRequest(
                f"req_{idx}", "144p", {"request_id": f"req_{idx}"}
            )
            for idx in range(3)
        ]
        starts = []
        lock = threading.Lock()

        class FakeResponse:
            status_code = 200
            text = "{}"

            def json(self):
                return {}

        def fake_post(_endpoint, *, json, timeout):
            del json, timeout
            with lock:
                starts.append(time.perf_counter())
            time.sleep(0.15)
            return FakeResponse()

        self.module.send_workload(
            server_url="http://127.0.0.1:30000",
            workload=workload,
            rate=20.0,
            timeout=10,
            max_inflight=3,
            post_fn=fake_post,
        )

        self.assertEqual(len(starts), 3)
        self.assertLess(starts[1] - starts[0], 0.12)
        self.assertLess(starts[2] - starts[1], 0.12)


if __name__ == "__main__":
    unittest.main()
