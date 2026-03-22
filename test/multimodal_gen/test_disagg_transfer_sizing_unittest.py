import unittest

_IMPORT_ERROR = None

try:
    import torch

    from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
        estimate_transfer_bytes,
    )
    from sglang.multimodal_gen.runtime.disaggregation.transport.manager import (
        DiffusionTransferManager,
    )
except Exception as exc:  # pragma: no cover - test is skipped when deps are absent.
    torch = None
    estimate_transfer_bytes = None
    DiffusionTransferManager = None
    _IMPORT_ERROR = exc


_SKIP_REASON = None
if torch is None or estimate_transfer_bytes is None or DiffusionTransferManager is None:
    _SKIP_REASON = f"torch runtime unavailable: {_IMPORT_ERROR}"


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON)
class TestDisaggTransferSizing(unittest.TestCase):
    def test_transfer_size_alignment_matches_scheduler_estimator(self):
        tensor_fields = {
            "latents": [
                torch.zeros(3, dtype=torch.float16),
                torch.zeros(7, dtype=torch.float16),
            ],
            "prompt_embeds": torch.zeros(5, dtype=torch.float32),
        }

        expected = 1536

        self.assertEqual(
            DiffusionTransferManager._estimate_transfer_size(tensor_fields),
            expected,
        )
        self.assertEqual(estimate_transfer_bytes(tensor_fields), expected)

    def test_transfer_size_estimator_handles_empty_payload(self):
        tensor_fields = {"latents": None}

        self.assertEqual(
            DiffusionTransferManager._estimate_transfer_size(tensor_fields),
            0,
        )
        self.assertEqual(estimate_transfer_bytes(tensor_fields), 0)


if __name__ == "__main__":
    unittest.main()
