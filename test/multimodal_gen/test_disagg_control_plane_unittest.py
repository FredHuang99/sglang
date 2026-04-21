import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.disaggregation.diffusion_server import (
    DiffusionServer,
    _TransferRequestState,
)
from sglang.multimodal_gen.runtime.disaggregation.dispatch_policy import (
    MaxFreeSlotsFirst,
    PoolDispatcher,
)
from sglang.multimodal_gen.runtime.disaggregation.request_state import (
    RequestState,
    TransferPhase,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import (
    RoleType,
    filter_modules_for_role,
)
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    SchedulerDisaggMixin,
    _is_skip_broadcast,
    _should_broadcast_encoder_idle_skip,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TransferPeerInfoMsg,
    TransferRegisterMsg,
    decode_transfer_msg,
    encode_transfer_msg,
    is_transfer_message,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class _FakeFrontend:
    def __init__(self):
        self.sent = []

    def send_multipart(self, frames):
        self.sent.append(frames)


class _EncoderLoopDummy(SchedulerDisaggMixin):
    def __init__(self, *, rank0: bool, messages=None):
        self.server_args = SimpleNamespace(
            sp_degree=1,
            tp_size=2,
            enable_cfg_parallel=False,
        )
        self.gpu_id = 0 if rank0 else 1
        self._disagg_role = RoleType.ENCODER
        self._running = True
        self._consecutive_error_count = 0
        self._max_consecutive_errors = 3
        self.broadcasts = []
        self.steps = []
        self.cleaned = False
        self._messages = list(messages or [])
        self._pool_work_pull = object()

    def _process_outbound_staging_retry_once(self):
        return False

    def _process_swap_out_queue_once(self):
        return False

    def _process_send_ready_queue_once(self):
        return False

    def _maybe_apply_pending_transfer_reconfigure(self):
        return False

    def _has_pending_outbound_staging_retry(self):
        return False

    def _try_recv_work_noblock(self):
        return None

    def _broadcast_to_all_ranks(self, data):
        if self.gpu_id == 0:
            self.broadcasts.append(data)
            if _is_skip_broadcast(data):
                self._running = False
            return data
        msg = self._messages.pop(0)
        self.broadcasts.append(msg)
        return msg

    def _disagg_encoder_step(self, send_tensors_fn, frames):
        del send_tensors_fn
        self.steps.append(frames)

    def _cleanup_disagg(self):
        self.cleaned = True


def _make_diffusion_server(*, timeout_s=600.0, downstream_wait_timeout_s=120.0):
    server = DiffusionServer(
        frontend_endpoint="inproc://frontend-test",
        encoder_work_endpoints=["inproc://encoder-work"],
        denoiser_work_endpoints=["inproc://denoiser-work"],
        decoder_work_endpoints=["inproc://decoder-work"],
        encoder_result_endpoint="inproc://encoder-result",
        denoiser_result_endpoint="inproc://denoiser-result",
        decoder_result_endpoint="inproc://decoder-result",
        timeout_s=timeout_s,
        downstream_wait_timeout_s=downstream_wait_timeout_s,
        max_slots_per_instance=1,
    )
    server._frontend = _FakeFrontend()
    return server


class TestDisaggControlPlane(unittest.TestCase):
    def test_transfer_register_round_trip_preserves_direct_connect_fields(self):
        msg = TransferRegisterMsg(
            role="encoder",
            instance_id=7,
            session_id="sess-1",
            pool_ptr=1234,
            pool_size=4096,
            control_endpoint="tcp://10.0.0.1:9001",
            work_endpoint="tcp://10.0.0.1:9000",
            rank0_only=True,
            role_device="cpu",
            preallocated_slots=[
                {
                    "offset": 512,
                    "size": 2048,
                    "slot_id": 3,
                    "addr": 1746,
                }
            ],
        )

        frames = encode_transfer_msg(msg)
        decoded = decode_transfer_msg(frames)

        self.assertTrue(is_transfer_message(frames))
        self.assertEqual(decoded["role"], "encoder")
        self.assertEqual(decoded["instance_id"], 7)
        self.assertEqual(decoded["control_endpoint"], "tcp://10.0.0.1:9001")
        self.assertEqual(decoded["work_endpoint"], "tcp://10.0.0.1:9000")
        self.assertTrue(decoded["rank0_only"])
        self.assertEqual(decoded["role_device"], "cpu")
        self.assertEqual(decoded["preallocated_slots"][0]["slot_id"], 3)

    def test_transfer_peer_info_round_trip_preserves_receiver_fields(self):
        msg = TransferPeerInfoMsg(
            request_id="req-1",
            dest_session_id="dst-session",
            dest_addr=987654,
            transfer_size=8192,
            receiver_role="decoder",
            receiver_instance=2,
            receiver_control_endpoint="tcp://10.0.0.3:9101",
            prealloc_slot_id=5,
        )

        decoded = decode_transfer_msg(encode_transfer_msg(msg))

        self.assertEqual(decoded["request_id"], "req-1")
        self.assertEqual(decoded["transfer_size"], 8192)
        self.assertEqual(decoded["receiver_role"], "decoder")
        self.assertEqual(decoded["receiver_instance"], 2)
        self.assertEqual(
            decoded["receiver_control_endpoint"],
            "tcp://10.0.0.3:9101",
        )
        self.assertEqual(decoded["prealloc_slot_id"], 5)

    def test_server_args_resolve_control_endpoint_and_role_device(self):
        args = object.__new__(ServerArgs)
        args.scheduler_port = 31000
        args.host = "0.0.0.0"
        args.disagg_p2p_hostname = "10.1.2.3"
        args.num_gpus = 0
        args.disagg_role_device = "auto"

        self.assertEqual(
            args.derive_pool_control_endpoint(),
            "tcp://0.0.0.0:31001",
        )
        self.assertEqual(
            args.derive_pool_control_advertised_endpoint(),
            "tcp://10.1.2.3:31001",
        )
        self.assertEqual(args.resolved_role_device(), "cpu")

        args.num_gpus = 4
        self.assertEqual(args.resolved_role_device(), "cuda")

        args.disagg_role_device = "cpu"
        self.assertEqual(args.resolved_role_device(), "cpu")

    def test_max_free_slots_policy_uses_explicit_capacity(self):
        policy = MaxFreeSlotsFirst(num_instances=3, max_slots_per_instance=4)
        self.assertEqual(policy.select(active_counts=[4, 1, 2]), 1)
        self.assertIsNone(policy.select_with_capacity([0, 0, 0]))

        dispatcher = PoolDispatcher(
            num_encoders=1,
            num_denoisers=3,
            num_decoders=1,
            policy_name="max_free_slots",
            max_slots_per_instance=4,
        )
        self.assertEqual(dispatcher.select_denoiser(active_counts=[4, 1, 2]), 1)

    def test_encoder_module_filtering_requires_decoder_opt_in(self):
        module_names = ["text_encoder", "vae", "transformer", "scheduler"]

        filtered_default = filter_modules_for_role(module_names, RoleType.ENCODER)
        filtered_with_decoder = filter_modules_for_role(
            module_names,
            RoleType.ENCODER,
            extra_allowed_modules={"vae"},
        )

        self.assertEqual(filtered_default, ["text_encoder", "scheduler"])
        self.assertEqual(
            filtered_with_decoder,
            ["text_encoder", "vae", "scheduler"],
        )

    def test_encoder_idle_skip_helpers(self):
        self.assertTrue(_is_skip_broadcast(("skip",)))
        self.assertFalse(_is_skip_broadcast(("encoder_work", [])))
        self.assertFalse(_should_broadcast_encoder_idle_skip(1.0, 1.005))
        self.assertTrue(_should_broadcast_encoder_idle_skip(1.0, 1.011))

    def test_encoder_rank0_broadcasts_idle_skip(self):
        dummy = _EncoderLoopDummy(rank0=True)

        dummy._disagg_encoder_rank0_event_loop()

        self.assertIn(("skip",), dummy.broadcasts)
        self.assertIsNone(dummy.broadcasts[-1])
        self.assertTrue(dummy.cleaned)

    def test_encoder_follower_ignores_idle_skip(self):
        dummy = _EncoderLoopDummy(rank0=False, messages=[("skip",), None])

        dummy._disagg_encoder_non_rank0_event_loop()

        self.assertEqual(dummy.steps, [])
        self.assertTrue(dummy.cleaned)

    def test_denoiser_done_without_decoder_payload_fails_request(self):
        server = _make_diffusion_server()
        request_id = "req-missing-decoder-payload"
        server._pending[request_id] = b"client"
        server._tracker.submit(request_id)
        server._tracker.transition(
            request_id, RequestState.ENCODER_RUNNING, encoder_instance=0
        )
        server._tracker.transition(request_id, RequestState.ENCODER_DONE)
        server._tracker.transition(request_id, RequestState.DENOISING_WAITING)
        server._tracker.transition(
            request_id, RequestState.DENOISING_RUNNING, denoiser_instance=0
        )
        server._encoder_free_slots[0] = 0
        server._denoiser_free_slots[0] = 0
        server._transfer_state[request_id] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            sender_instance=0,
            sender_slot_released=True,
            receiver_role=RoleType.DENOISER.value,
            receiver_instance=0,
            transfer_phase=TransferPhase.RUNNING_DOWNSTREAM,
        )

        server._handle_transfer_done(
            {"request_id": request_id, "staged_for_decoder": False},
            RoleType.DENOISER,
        )

        self.assertIsNone(server._tracker.get(request_id))
        self.assertNotIn(request_id, server._pending)
        self.assertNotIn(request_id, server._transfer_state)
        self.assertEqual(server._denoiser_free_slots[0], 1)
        self.assertEqual(len(server._frontend.sent), 1)

    def test_global_timeout_cleans_transfer_state_and_returns_error(self):
        server = _make_diffusion_server(timeout_s=0.1)
        request_id = "req-global-timeout"
        server._pending[request_id] = b"client"
        record = server._tracker.submit(request_id)
        server._tracker.transition(
            request_id, RequestState.ENCODER_RUNNING, encoder_instance=0
        )
        server._tracker.transition(request_id, RequestState.ENCODER_DONE)
        server._tracker.transition(request_id, RequestState.DENOISING_WAITING)
        server._tracker.transition(
            request_id, RequestState.DENOISING_RUNNING, denoiser_instance=0
        )
        record.submit_time -= 1.0
        server._encoder_free_slots[0] = 0
        server._denoiser_free_slots[0] = 0
        server._transfer_state[request_id] = _TransferRequestState(
            sender_role=RoleType.ENCODER.value,
            sender_instance=0,
            sender_slot_released=True,
            receiver_role=RoleType.DENOISER.value,
            receiver_instance=0,
            transfer_phase=TransferPhase.RUNNING_DOWNSTREAM,
        )

        server._handle_timeouts()

        self.assertIsNone(server._tracker.get(request_id))
        self.assertNotIn(request_id, server._pending)
        self.assertNotIn(request_id, server._transfer_state)
        self.assertEqual(server._denoiser_free_slots[0], 1)
        self.assertEqual(len(server._frontend.sent), 1)


if __name__ == "__main__":
    unittest.main()
