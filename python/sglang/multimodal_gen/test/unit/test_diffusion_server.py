# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DiffusionServer pool-based pipeline orchestrator."""

import unittest
from unittest.mock import MagicMock

from sglang.multimodal_gen.runtime.disaggregation.diffusion_server import (
    DiffusionServer,
    _TransferRequestState,
)
from sglang.multimodal_gen.runtime.disaggregation.request_state import (
    RequestState,
    TransferPhase,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TransferAllocAcceptedMsg,
    TransferAllocRejectMsg,
    TransferPushedMsg,
    TransferRegisterMsg,
    TransferStagedMsg,
    decode_transfer_msg,
    encode_transfer_msg,
)


class TestDiffusionServerInit(unittest.TestCase):
    def test_basic_init(self):
        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19900",
            encoder_work_endpoints=["tcp://127.0.0.1:19901"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19902"],
            decoder_work_endpoints=["tcp://127.0.0.1:19903"],
            encoder_result_endpoint="tcp://127.0.0.1:19904",
            denoiser_result_endpoint="tcp://127.0.0.1:19905",
            decoder_result_endpoint="tcp://127.0.0.1:19906",
            max_slots_per_instance=3,
        )
        self.assertEqual(server._encoder_free_slots, [3])
        self.assertEqual(server._denoiser_free_slots, [3])
        self.assertEqual(server._decoder_free_slots, [3])


class TestDiffusionServerTransferProtocol(unittest.TestCase):
    def setUp(self):
        self.server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19910",
            encoder_work_endpoints=["tcp://127.0.0.1:19911"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19912"],
            decoder_work_endpoints=["tcp://127.0.0.1:19913"],
            encoder_result_endpoint="tcp://127.0.0.1:19914",
            denoiser_result_endpoint="tcp://127.0.0.1:19915",
            decoder_result_endpoint="tcp://127.0.0.1:19916",
            max_slots_per_instance=2,
        )
        self.server._encoder_pushes = [MagicMock()]
        self.server._denoiser_pushes = [MagicMock()]
        self.server._decoder_pushes = [MagicMock()]

    def _submit_running_request(self, request_id: str, state: RequestState):
        record = self.server._tracker.submit(request_id)
        if state == RequestState.ENCODER_RUNNING:
            record.encoder_instance = 0
        elif state in (
            RequestState.DENOISING_RUNNING,
            RequestState.DENOISING_DONE,
            RequestState.DENOISING_WAITING,
        ):
            record.encoder_instance = 0
            record.denoiser_instance = 0
        elif state in (RequestState.DECODER_RUNNING, RequestState.DECODER_WAITING):
            record.encoder_instance = 0
            record.denoiser_instance = 0
            record.decoder_instance = 0
        record.state = state

    def test_transfer_register_tracks_host_meta_and_prealloc(self):
        reg_msg = TransferRegisterMsg(
            role="denoiser",
            instance_id=0,
            session_id="den-session-0",
            pool_ptr=0x7F000000,
            pool_size=16 * 1024 * 1024,
            meta_pool_ptr=0x8F000000,
            meta_pool_size=128 * 1024,
            control_endpoint="tcp://den-ctrl",
            host_id="host-a",
            supports_local_copy=True,
            data_shm_name="data-shm",
            meta_shm_name="meta-shm",
            preallocated_slots=[
                {
                    "slot_id": 4,
                    "offset": 256,
                    "size": 4096,
                    "addr": 0x7F000100,
                    "meta_offset": 128,
                    "meta_size": 2048,
                    "meta_addr": 0x8F000080,
                }
            ],
        )
        self.server._handle_transfer_result(encode_transfer_msg(reg_msg), RoleType.DENOISER)

        peer = self.server._denoiser_peers[0]
        self.assertEqual(peer["control_endpoint"], "tcp://den-ctrl")
        self.assertEqual(peer["host_id"], "host-a")
        self.assertTrue(peer["supports_local_copy"])
        self.assertEqual(peer["meta_pool_ptr"], 0x8F000000)
        self.assertEqual(peer["free_preallocated_slots"][0]["slot_id"], 4)
        self.assertEqual(peer["free_preallocated_slots"][0]["meta_size"], 2048)

    def test_transfer_staged_dispatches_alloc_with_meta_and_host(self):
        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferRegisterMsg(
                    role="encoder",
                    instance_id=0,
                    session_id="enc-session-0",
                    pool_ptr=0x1000,
                    pool_size=16 * 1024 * 1024,
                    meta_pool_ptr=0x1800,
                    meta_pool_size=128 * 1024,
                    control_endpoint="tcp://enc-ctrl",
                    host_id="host-a",
                )
            ),
            RoleType.ENCODER,
        )
        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferRegisterMsg(
                    role="denoiser",
                    instance_id=0,
                    session_id="den-session-0",
                    pool_ptr=0x2000,
                    pool_size=16 * 1024 * 1024,
                    meta_pool_ptr=0x2800,
                    meta_pool_size=128 * 1024,
                    control_endpoint="tcp://den-ctrl",
                    host_id="host-a",
                    preallocated_slots=[
                        {
                            "slot_id": 1,
                            "offset": 512,
                            "size": 4096,
                            "addr": 0x2000 + 512,
                            "meta_offset": 128,
                            "meta_size": 2048,
                            "meta_addr": 0x2800 + 128,
                        }
                    ],
                )
            ),
            RoleType.DENOISER,
        )
        self._submit_running_request("r1", RequestState.ENCODER_RUNNING)

        staged_msg = TransferStagedMsg(
            request_id="r1",
            data_size=4096,
            meta_size=2048,
            session_id="enc-session-0",
            pool_ptr=0x1000,
            slot_offset=0,
            meta_pool_ptr=0x1800,
            meta_slot_offset=64,
        )
        self.server._handle_transfer_result(encode_transfer_msg(staged_msg), RoleType.ENCODER)
        self.server._drain_denoiser_tta()

        sent_frames = self.server._denoiser_pushes[0].send_multipart.call_args[0][0]
        alloc_msg = decode_transfer_msg(sent_frames)
        self.assertEqual(alloc_msg["msg_type"], "transfer_alloc")
        self.assertEqual(alloc_msg["source_control_endpoint"], "tcp://enc-ctrl")
        self.assertEqual(alloc_msg["source_host_id"], "host-a")
        self.assertEqual(alloc_msg["meta_size"], 2048)
        self.assertEqual(alloc_msg["preallocated_slot"]["slot_id"], 1)
        self.assertEqual(
            self.server._tracker.get("r1").state,
            RequestState.DENOISING_WAITING,
        )

    def test_transfer_staged_keeps_encoder_slot_busy_until_push_and_starts_wait_timer(self):
        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferRegisterMsg(
                    role="encoder",
                    instance_id=0,
                    session_id="enc-session-0",
                    pool_ptr=0x1000,
                    pool_size=16 * 1024 * 1024,
                    meta_pool_ptr=0x1800,
                    meta_pool_size=128 * 1024,
                    control_endpoint="tcp://enc-ctrl",
                    host_id="host-a",
                )
            ),
            RoleType.ENCODER,
        )
        self._submit_running_request("r-stage", RequestState.ENCODER_RUNNING)
        self.server._encoder_free_slots[0] = 0
        staged_msg = TransferStagedMsg(
            request_id="r-stage",
            data_size=4096,
            meta_size=2048,
            session_id="enc-session-0",
            pool_ptr=0x1000,
            slot_offset=0,
            meta_pool_ptr=0x1800,
            meta_slot_offset=64,
        )

        self.server._handle_transfer_result(encode_transfer_msg(staged_msg), RoleType.ENCODER)

        self.assertEqual(self.server._encoder_free_slots[0], 0)
        self.assertEqual(
            self.server._tracker.get("r-stage").state,
            RequestState.DENOISING_WAITING,
        )
        self.assertIsNotNone(self.server._transfer_state["r-stage"].downstream_wait_since)
        self.assertEqual(
            self.server._transfer_state["r-stage"].transfer_phase,
            TransferPhase.WAITING_FOR_DOWNSTREAM_SLOT,
        )

    def test_transfer_pushed_releases_sender_slot_once_and_starts_running(self):
        self._submit_running_request("r-pushed", RequestState.DENOISING_WAITING)
        self.server._tracker.update_instances("r-pushed", denoiser_instance=0)
        self.server._encoder_free_slots[0] = 0
        self.server._transfer_state["r-pushed"] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            sender_instance=0,
            receiver_instance=0,
            sender_slot_released=False,
            alloc_accepted=True,
        )

        pushed = encode_transfer_msg(TransferPushedMsg(request_id="r-pushed", success=True))
        self.server._handle_transfer_result(pushed, RoleType.ENCODER)
        self.server._handle_transfer_result(pushed, RoleType.ENCODER)

        self.assertEqual(self.server._encoder_free_slots[0], 1)
        self.assertEqual(
            self.server._tracker.get("r-pushed").state,
            RequestState.DENOISING_RUNNING,
        )

    def test_fatal_alloc_reject_releases_sender_slot(self):
        self._submit_running_request("r-fatal", RequestState.DENOISING_WAITING)
        self.server._pending["r-fatal"] = b"client"
        self.server._frontend = MagicMock()
        self.server._send_abort = MagicMock()
        self.server._encoder_free_slots[0] = 0
        self.server._tracker.update_instances("r-fatal", denoiser_instance=0)
        self.server._transfer_state["r-fatal"] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            sender_instance=0,
            receiver_instance=0,
            sender_control_endpoint="tcp://enc-ctrl",
            sender_slot_released=False,
            downstream_wait_since=1.0,
        )

        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferAllocRejectMsg(
                    request_id="r-fatal",
                    receiver_role=RoleType.DENOISER.value,
                    receiver_instance=0,
                    retryable=False,
                    reason="fatal-busy",
                )
            ),
            RoleType.DENOISER,
        )

        self.assertEqual(self.server._encoder_free_slots[0], 1)
        self.server._send_abort.assert_called_once()
        self.assertIsNone(self.server._tracker.get("r-fatal"))

    def test_retryable_alloc_reject_requeues_request(self):
        self._submit_running_request("r-retry", RequestState.DENOISING_WAITING)
        self.server._tracker.update_instances("r-retry", denoiser_instance=0)
        self.server._denoiser_free_slots[0] = 0
        p2p = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            sender_instance=0,
            receiver_instance=0,
            downstream_wait_since=1.0,
        )
        self.server._transfer_state["r-retry"] = p2p

        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferAllocRejectMsg(
                    request_id="r-retry",
                    receiver_role=RoleType.DENOISER.value,
                    receiver_instance=0,
                    retryable=True,
                    reason="busy",
                )
            ),
            RoleType.DENOISER,
        )

        self.assertIn("r-retry", self.server._transfer_state)
        self.assertEqual(len(self.server._denoiser_tta), 1)
        self.assertEqual(self.server._denoiser_tta[0].request_id, "r-retry")
        self.assertEqual(
            self.server._tracker.get("r-retry").state,
            RequestState.DENOISING_WAITING,
        )

    def test_alloc_accepted_stops_downstream_wait_timer(self):
        self._submit_running_request("r-accept", RequestState.DENOISING_WAITING)
        p2p = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            sender_instance=0,
            receiver_instance=0,
            downstream_wait_since=123.0,
        )
        self.server._transfer_state["r-accept"] = p2p

        self.server._handle_transfer_result(
            encode_transfer_msg(
                TransferAllocAcceptedMsg(
                    request_id="r-accept",
                    receiver_role=RoleType.DENOISER.value,
                    receiver_instance=0,
                )
            ),
            RoleType.DENOISER,
        )

        self.assertTrue(self.server._transfer_state["r-accept"].alloc_accepted)
        self.assertIsNone(self.server._transfer_state["r-accept"].downstream_wait_since)
        self.assertEqual(
            self.server._transfer_state["r-accept"].transfer_phase,
            TransferPhase.SENDING,
        )

    def test_alloc_result_timeout_requeues_request_instead_of_failing(self):
        self._submit_running_request("r-alloc-timeout", RequestState.DENOISING_WAITING)
        self.server._tracker.update_instances("r-alloc-timeout", denoiser_instance=0)
        self.server._denoiser_free_slots[0] = 0
        self.server._downstream_wait_timeout_s = 100.0
        self.server._alloc_result_timeout_s = 1.0
        self.server._transfer_state["r-alloc-timeout"] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            sender_instance=0,
            receiver_instance=0,
            receiver_pool_ptr=0x3000,
            receiver_slot_offset=256,
            receiver_slot_size=4096,
            receiver_meta_pool_ptr=0x3800,
            receiver_meta_slot_offset=64,
            receiver_meta_slot_size=2048,
            meta_size=2048,
            prealloc_slot_id=7,
            transfer_phase=TransferPhase.WAITING_ALLOC_RESULT,
            handoff_started_at=0.0,
            phase_started_at=0.0,
            downstream_wait_since=0.0,
        )
        self.server._denoiser_peers[0] = {
            "free_preallocated_slots": [],
        }

        with unittest.mock.patch(
            "sglang.multimodal_gen.runtime.disaggregation.diffusion_server.time.monotonic",
            return_value=10.0,
        ):
            self.server._handle_timeouts()

        self.assertIn("r-alloc-timeout", self.server._transfer_state)
        self.assertEqual(len(self.server._denoiser_tta), 1)
        self.assertEqual(self.server._denoiser_free_slots[0], 1)
        self.assertEqual(
            self.server._transfer_state["r-alloc-timeout"].transfer_phase,
            TransferPhase.WAITING_FOR_DOWNSTREAM_SLOT,
        )

    def test_downstream_wait_timeout_aborts_sender_only_and_times_out(self):
        self._submit_running_request("r-timeout", RequestState.DENOISING_WAITING)
        self.server._pending["r-timeout"] = b"client"
        self.server._frontend = MagicMock()
        self.server._send_abort = MagicMock()
        self.server._downstream_wait_timeout_s = 1.0
        self.server._encoder_free_slots[0] = 0
        self.server._transfer_state["r-timeout"] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            sender_instance=0,
            sender_control_endpoint="tcp://enc-ctrl",
            downstream_wait_since=0.0,
        )

        with unittest.mock.patch(
            "sglang.multimodal_gen.runtime.disaggregation.diffusion_server.time.monotonic",
            return_value=10.0,
        ):
            self.server._handle_timeouts()

        self.server._send_abort.assert_called_once()
        _args, kwargs = self.server._send_abort.call_args
        self.assertTrue(kwargs["to_sender"])
        self.assertFalse(kwargs["to_receiver"])
        self.assertEqual(self.server._encoder_free_slots[0], 1)
        self.assertIsNone(self.server._tracker.get("r-timeout"))

    def test_denoiser_done_keeps_slot_busy_for_decoder_handoff_and_enqueues_once(self):
        self._submit_running_request("r-done", RequestState.DENOISING_RUNNING)
        self.server._denoiser_peers[0] = {
            "control_endpoint": "tcp://den-ctrl",
            "host_id": "host-a",
            "free_preallocated_slots": [],
        }
        self.server._decoder_peers[0] = {
            "control_endpoint": "tcp://dec-ctrl",
            "host_id": "host-a",
            "free_preallocated_slots": [],
        }
        self.server._denoiser_free_slots[0] = 0
        self.server._transfer_state["r-done"] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            receiver_role=RoleType.DENOISER.value,
            receiver_instance=0,
            receiver_pool_ptr=0x3000,
            receiver_slot_offset=256,
            receiver_slot_size=4096,
            receiver_meta_pool_ptr=0x3800,
            receiver_meta_slot_offset=64,
            receiver_meta_slot_size=2048,
            meta_size=2048,
            prealloc_slot_id=7,
        )

        done_msg = {
            "request_id": "r-done",
            "staged_for_decoder": True,
            "session_id": "den-session",
            "pool_ptr": 0x5000,
            "slot_offset": 128,
            "meta_pool_ptr": 0x5800,
            "meta_slot_offset": 32,
            "data_size": 2048,
            "meta_size": 1024,
        }

        self.server._handle_transfer_done(done_msg, RoleType.DENOISER)
        self.server._handle_transfer_done(done_msg, RoleType.DENOISER)

        self.assertEqual(self.server._denoiser_free_slots[0], 0)
        self.assertEqual(len(self.server._denoiser_peers[0]["free_preallocated_slots"]), 1)
        self.assertEqual(
            self.server._denoiser_peers[0]["free_preallocated_slots"][0]["meta_size"],
            2048,
        )
        self.assertEqual(len(self.server._decoder_tta), 1)
        self.assertEqual(
            self.server._tracker.get("r-done").state,
            RequestState.DECODER_WAITING,
        )
        self.assertFalse(self.server._transfer_state["r-done"].sender_slot_released)

    def test_second_hop_push_releases_denoiser_slot_once(self):
        self._submit_running_request("r-second-push", RequestState.DECODER_WAITING)
        self.server._tracker.update_instances("r-second-push", denoiser_instance=0, decoder_instance=0)
        self.server._denoiser_free_slots[0] = 0
        self.server._transfer_state["r-second-push"] = _TransferRequestState(
            sender_role=RoleType.DENOISER.value,
            receiver_role=RoleType.DECODER.value,
            sender_instance=0,
            receiver_instance=0,
            sender_slot_released=False,
            alloc_accepted=True,
        )

        pushed = encode_transfer_msg(
            TransferPushedMsg(request_id="r-second-push", success=True)
        )
        self.server._handle_transfer_result(pushed, RoleType.DENOISER)
        self.server._handle_transfer_result(pushed, RoleType.DENOISER)

        self.assertEqual(self.server._denoiser_free_slots[0], 1)
        self.assertEqual(
            self.server._tracker.get("r-second-push").state,
            RequestState.DECODER_RUNNING,
        )


if __name__ == "__main__":
    unittest.main()
