# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Scheduler transfer integration."""

import pickle
import queue
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    _PendingInboundTransfer,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    TransferMetaBuffer,
    TransferTensorBuffer,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    MockTransferEngine,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.manager import (
    DiffusionTransferManager,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TransferAllocMsg,
    TransferMsgType,
    decode_transfer_msg,
    encode_transfer_msg,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler


class TestSchedulerTransferFrameDetection(unittest.TestCase):
    def test_transfer_frames_detected(self):
        frames = encode_transfer_msg(
            TransferAllocMsg(request_id="r1", data_size=1024, meta_size=128)
        )
        self.assertTrue(Scheduler._is_transfer_frames(frames))

    def test_non_transfer_frames_not_detected(self):
        self.assertFalse(Scheduler._is_transfer_frames([b"some_data", b"more_data"]))


class _SchedulerHarness:
    @staticmethod
    def make(role: RoleType) -> Scheduler:
        scheduler = object.__new__(Scheduler)
        scheduler._disagg_role = role
        scheduler._disagg_metrics = None
        scheduler._pool_result_push = MagicMock()
        scheduler._preallocated_slots = {}
        scheduler._control_queue = queue.Queue()
        scheduler._transferring_queue = queue.Queue()
        scheduler._prefetch_queue = scheduler._transferring_queue
        scheduler._swapping_queue = queue.Queue()
        scheduler._compute_ready_queue = queue.Queue()
        scheduler._swap_out_queue = deque()
        scheduler._send_ready_queue = deque()
        scheduler._swap_in_stream = None
        scheduler._compute_stream = None
        scheduler._swap_out_stream = None
        scheduler._pending_transfer_reconfigure = None
        scheduler._transfer_reconfigured = False
        scheduler._warmup_inbound_sizes = {}
        scheduler._aborted_request_ids = {}
        scheduler._running = True
        scheduler.gpu_id = 0
        scheduler.worker = SimpleNamespace(
            local_rank=0,
            pipeline=SimpleNamespace(get_module=lambda name: None),
        )
        scheduler.server_args = SimpleNamespace(
            disagg_instance_id=1,
            pool_control_advertised_endpoint="tcp://receiver-ctrl",
            pool_control_endpoint="tcp://receiver-ctrl",
            sp_degree=1,
            tp_size=1,
            enable_cfg_parallel=False,
            resolved_role_device=lambda: "cpu",
        )
        return scheduler


class _TrackedStreamContext:
    def __init__(self, state, stream):
        self._state = state
        self._stream = stream

    def __enter__(self):
        self._state["active"] = True
        self._state["stream"] = self._stream
        return self

    def __exit__(self, exc_type, exc, tb):
        self._state["active"] = False
        return False


class TestSchedulerTransferAlloc(unittest.TestCase):
    def setUp(self):
        MockTransferEngine.reset()
        self.addCleanup(MockTransferEngine.reset)
        self.engine = MockTransferEngine(session_id="receiver-session")
        self.buffer = TransferTensorBuffer(pool_size=1 * 1024 * 1024, role_name="test")
        self.meta_buffer = TransferMetaBuffer(slot_count=2, slot_size=64 * 1024, role_name="test")
        self.tm = DiffusionTransferManager(
            engine=self.engine,
            buffer=self.buffer,
            meta_buffer=self.meta_buffer,
            host_id="host-a",
        )
        self.addCleanup(self.tm.cleanup)
        self.scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        self.scheduler._transfer_manager = self.tm
        self.tm.send_direct_message = MagicMock()

    def test_alloc_sends_peer_info_to_upstream_with_meta_and_local_copy(self):
        msg = {
            "msg_type": TransferMsgType.ALLOC,
            "request_id": "req-alloc-1",
            "data_size": 4096,
            "meta_size": 2048,
            "source_control_endpoint": "tcp://upstream-ctrl",
            "source_host_id": "host-a",
        }

        self.scheduler._handle_transfer_alloc(msg)

        self.assertIsNotNone(self.tm.get_receive_slot_addr("req-alloc-1"))
        self.assertIsNotNone(self.tm.get_receive_meta_addr("req-alloc-1"))
        self.tm.send_direct_message.assert_called_once()
        endpoint, peer_msg = self.tm.send_direct_message.call_args[0]
        self.assertEqual(endpoint, "tcp://upstream-ctrl")
        self.assertEqual(peer_msg.msg_type, TransferMsgType.PEER_INFO)
        self.assertEqual(peer_msg.request_id, "req-alloc-1")
        self.assertEqual(peer_msg.dest_session_id, self.engine.session_id)
        self.assertEqual(peer_msg.receiver_control_endpoint, "tcp://receiver-ctrl")
        self.assertEqual(peer_msg.meta_transfer_size, 2048)
        self.assertEqual(peer_msg.receiver_host_id, "host-a")
        self.assertTrue(peer_msg.receiver_supports_local_copy)
        self.assertEqual(peer_msg.dest_shm_name, self.tm.data_shm_name)
        self.assertEqual(peer_msg.meta_dest_shm_name, self.tm.meta_shm_name)
        self.scheduler._pool_result_push.send_multipart.assert_called_once()
        sent_frames = self.scheduler._pool_result_push.send_multipart.call_args[0][0]
        accepted_msg = decode_transfer_msg(sent_frames)
        self.assertEqual(accepted_msg["msg_type"], TransferMsgType.ALLOC_ACCEPTED)
        self.assertEqual(accepted_msg["receiver_role"], RoleType.DENOISER.value)
        self.assertEqual(accepted_msg["request_id"], "req-alloc-1")

    def test_missing_source_control_endpoint_reports_non_retryable_alloc_reject(self):
        msg = {
            "msg_type": TransferMsgType.ALLOC,
            "request_id": "req-alloc-2",
            "data_size": 4096,
            "meta_size": 512,
        }

        self.scheduler._handle_transfer_alloc(msg)

        self.assertIsNone(self.tm.get_receive_slot_addr("req-alloc-2"))
        self.tm.send_direct_message.assert_not_called()
        self.scheduler._pool_result_push.send_multipart.assert_called_once()
        sent_frames = self.scheduler._pool_result_push.send_multipart.call_args[0][0]
        reply = decode_transfer_msg(sent_frames)
        self.assertEqual(reply["msg_type"], TransferMsgType.ALLOC_REJECT)
        self.assertFalse(reply["retryable"])
        self.assertEqual(reply["reason"], "missing source control endpoint")
        self.assertEqual(reply["request_id"], "req-alloc-2")

    def test_allocate_receive_slot_failure_reports_retryable_alloc_reject(self):
        self.tm.allocate_receive_slot = MagicMock(return_value=None)
        msg = {
            "msg_type": TransferMsgType.ALLOC,
            "request_id": "req-alloc-3",
            "data_size": 4096,
            "meta_size": 512,
            "source_control_endpoint": "tcp://upstream-ctrl",
            "source_host_id": "host-a",
        }

        self.scheduler._handle_transfer_alloc(msg)

        self.assertIsNone(self.tm.get_receive_slot_addr("req-alloc-3"))
        self.tm.send_direct_message.assert_not_called()
        self.scheduler._pool_result_push.send_multipart.assert_called_once()
        sent_frames = self.scheduler._pool_result_push.send_multipart.call_args[0][0]
        reply = decode_transfer_msg(sent_frames)
        self.assertEqual(reply["msg_type"], TransferMsgType.ALLOC_REJECT)
        self.assertTrue(reply["retryable"])
        self.assertEqual(reply["reason"], "receiver failed to allocate slot")
        self.assertEqual(reply["request_id"], "req-alloc-3")


class TestSchedulerPrefetchQueues(unittest.TestCase):
    def test_direct_callbacks_queue_work_without_inline_execution(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler._handle_transfer_ready = MagicMock()
        scheduler._handle_transfer_failed = MagicMock()

        ready_msg = {"msg_type": TransferMsgType.READY, "request_id": "ready-1"}
        failed_msg = {"msg_type": TransferMsgType.FAILED, "request_id": "fail-1"}

        scheduler._on_direct_transfer_ready(ready_msg)
        scheduler._on_direct_transfer_failed(failed_msg)

        self.assertEqual(scheduler._transferring_queue.get_nowait(), ready_msg)
        self.assertEqual(scheduler._control_queue.get_nowait(), failed_msg)
        scheduler._handle_transfer_ready.assert_not_called()
        scheduler._handle_transfer_failed.assert_not_called()

    def test_encoder_ignores_ready_callback(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler._transferring_queue = None
        scheduler._handle_transfer_ready = MagicMock()

        scheduler._on_direct_transfer_ready(
            {"msg_type": TransferMsgType.READY, "request_id": "ignored"}
        )

        scheduler._handle_transfer_ready.assert_not_called()

    def test_control_queue_is_serviced_before_transferring_queue(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        order = []

        scheduler._handle_transfer_alloc = (
            lambda msg: order.append(("alloc", msg["request_id"]))
        )
        scheduler._prefetch_transfer_ready = lambda msg: (
            order.append(("ready", msg["request_id"]))
            or _PendingInboundTransfer(
                request_id=msg["request_id"],
                role_name="DENOISER",
                scalar_fields={},
                tensors={},
                load_event=None,
                prealloc_slot_id=None,
            )
        )

        scheduler._control_queue.put(
            {"msg_type": TransferMsgType.ALLOC, "request_id": "alloc-first"}
        )
        scheduler._transferring_queue.put(
            {"msg_type": TransferMsgType.READY, "request_id": "ready-second"}
        )

        self.assertTrue(scheduler._process_transfer_control_queue())
        self.assertTrue(scheduler._process_prefetch_queue_once())
        self.assertEqual(order, [("alloc", "alloc-first"), ("ready", "ready-second")])
        self.assertEqual(
            scheduler._swapping_queue.get_nowait().request_id, "ready-second"
        )

    def test_swapping_queue_only_advances_ready_loads(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)

        class _FakeEvent:
            def __init__(self):
                self.ready = False

            def query(self):
                return self.ready

        event = _FakeEvent()
        item = _PendingInboundTransfer(
            request_id="r1",
            role_name="DENOISER",
            scalar_fields={},
            tensors={},
            load_event=event,
            prealloc_slot_id=None,
        )
        scheduler._swapping_queue.put(item)
        self.assertFalse(scheduler._process_swapping_queue_once())
        self.assertTrue(scheduler._compute_ready_queue.empty())
        event.ready = True
        self.assertTrue(scheduler._process_swapping_queue_once())
        self.assertEqual(scheduler._compute_ready_queue.get_nowait().request_id, "r1")


class TestSchedulerTransferReady(unittest.TestCase):
    def setUp(self):
        self.scheduler = _SchedulerHarness.make(RoleType.DECODER)
        self.scheduler._prefetch_transfer_ready = MagicMock(
            return_value=_PendingInboundTransfer(
                request_id="ready-direct",
                role_name="DECODER",
                scalar_fields={},
                tensors={},
                load_event=None,
                prealloc_slot_id=None,
            )
        )
        self.scheduler._wait_transfer_event_on_compute_stream = MagicMock()
        self.scheduler._run_prefetched_compute_item = MagicMock()
        self.scheduler._transferring_queue = None

    def test_handle_transfer_ready_reuses_prefetched_compute_path(self):
        msg = {
            "msg_type": TransferMsgType.READY,
            "request_id": "ready-direct",
        }

        self.scheduler._handle_transfer_ready(msg)

        self.scheduler._wait_transfer_event_on_compute_stream.assert_called_once()
        self.scheduler._run_prefetched_compute_item.assert_called_once()
        item = self.scheduler._run_prefetched_compute_item.call_args[0][0]
        self.assertEqual(item.request_id, "ready-direct")


class TestSchedulerTransferStreams(unittest.TestCase):
    def test_wait_transfer_event_on_compute_stream_binds_event_to_compute_stream(self):
        scheduler = _SchedulerHarness.make(RoleType.DECODER)
        scheduler.server_args.resolved_role_device = lambda: "cuda"
        scheduler._compute_stream = object()
        stream_state = {"active": False, "stream": None}
        current_stream = MagicMock()

        def _wait_event(event):
            self.assertTrue(stream_state["active"])
            self.assertIs(stream_state["stream"], scheduler._compute_stream)

        current_stream.wait_event.side_effect = _wait_event

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.torch.cuda.stream",
            side_effect=lambda stream: _TrackedStreamContext(stream_state, stream),
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.torch.cuda.current_stream",
            return_value=current_stream,
        ):
            scheduler._wait_transfer_event_on_compute_stream(object())

        current_stream.wait_event.assert_called_once()

    def test_prefetch_uses_swap_in_stream(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler._swap_in_stream = object()
        scheduler._transfer_manager = MagicMock()
        scheduler._transfer_manager.load_transfer_async.return_value = ({}, {}, None)

        scheduler._prefetch_transfer_ready({"request_id": "stream-in-1"})

        scheduler._transfer_manager.load_transfer_async.assert_called_once_with(
            "stream-in-1",
            device="cpu",
            stream=scheduler._swap_in_stream,
        )

    def test_encoder_staging_uses_swap_out_stream(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler._swap_out_stream = object()
        scheduler._transfer_manager = MagicMock()
        scheduler._transfer_manager.stage_tensors_async.return_value = (
            SimpleNamespace(
                transfer_size=1,
                meta_size=1,
                scalar_fields={},
                slot=None,
                meta_slot=None,
            ),
            None,
        )

        scheduler._disagg_encoder_transfer_stage(
            "stream-out-1",
            {"latents": torch.randn(1, 4, 4, 4)},
            {"request_id": "stream-out-1"},
        )

        scheduler._transfer_manager.stage_tensors_async.assert_called_once_with(
            request_id="stream-out-1",
            tensor_fields=unittest.mock.ANY,
            scalar_fields={"request_id": "stream-out-1"},
            stream=scheduler._swap_out_stream,
        )

    def test_encoder_forward_runs_on_compute_stream(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler.server_args.resolved_role_device = lambda: "cuda"
        scheduler._compute_stream = object()
        scheduler._swap_out_stream = object()
        scheduler._transfer_manager = MagicMock()
        scheduler._transfer_manager.stage_tensors_async.return_value = (
            SimpleNamespace(
                transfer_size=1,
                meta_size=1,
                scalar_fields={},
                slot=None,
                meta_slot=None,
            ),
            None,
        )
        stream_state = {"active": False, "stream": None}

        def _forward(batch, return_req=False):
            self.assertTrue(stream_state["active"])
            self.assertIs(stream_state["stream"], scheduler._compute_stream)
            return batch[0]

        scheduler.worker.execute_forward = MagicMock(side_effect=_forward)
        req = Req(request_id="enc-compute")
        frames = [b"enc-compute", pickle.dumps([req])]

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.torch.cuda.stream",
            side_effect=lambda stream: _TrackedStreamContext(stream_state, stream),
        ):
            scheduler._disagg_encoder_step(MagicMock(), frames=frames)

        scheduler.worker.execute_forward.assert_called_once()
        scheduler._transfer_manager.stage_tensors_async.assert_called_once()

    def test_encoder_abort_after_stage_cleans_local_transfer_state(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler._transfer_manager = MagicMock()
        scheduler._transfer_manager.stage_tensors_async.return_value = (
            SimpleNamespace(
                transfer_size=1,
                meta_size=1,
                scalar_fields={},
                slot=None,
                meta_slot=None,
            ),
            None,
        )
        scheduler._is_request_aborted = MagicMock(return_value=True)
        scheduler._warmup_inbound_sizes["enc-abort"] = (32, 16)
        scheduler.worker.execute_forward = MagicMock(
            return_value=Req(request_id="enc-abort")
        )
        frames = [b"enc-abort", pickle.dumps([Req(request_id="enc-abort")])]

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.extract_transfer_fields",
            return_value=({"latents": torch.randn(1, 1)}, {"request_id": "enc-abort"}),
        ):
            scheduler._disagg_encoder_step(MagicMock(), frames=frames)

        scheduler._transfer_manager.abort_request.assert_called_once_with("enc-abort")
        self.assertNotIn("enc-abort", scheduler._warmup_inbound_sizes)

    def test_denoiser_abort_after_stage_cleans_local_transfer_state(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler._transfer_manager = MagicMock()
        scheduler._transfer_manager.stage_tensors_async.return_value = (
            SimpleNamespace(
                transfer_size=1,
                meta_size=1,
                scalar_fields={},
                slot=None,
                meta_slot=None,
            ),
            None,
        )
        scheduler._enqueue_outbound_transfer = MagicMock()
        scheduler._is_request_aborted = MagicMock(side_effect=[False, True])
        scheduler._warmup_inbound_sizes["den-abort"] = (64, 32)
        scheduler.worker.execute_forward = MagicMock(
            return_value=Req(request_id="den-abort")
        )

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.extract_transfer_fields",
            return_value=({"latents": torch.randn(1, 1)}, {"request_id": "den-abort"}),
        ):
            scheduler._disagg_denoiser_compute(
                Req(request_id="den-abort"),
                "den-abort",
                "DENOISER",
            )

        scheduler._transfer_manager.abort_request.assert_called_once_with("den-abort")
        scheduler._enqueue_outbound_transfer.assert_not_called()
        self.assertNotIn("den-abort", scheduler._warmup_inbound_sizes)

    def test_decoder_forward_runs_on_compute_stream_and_waits_before_send(self):
        scheduler = _SchedulerHarness.make(RoleType.DECODER)
        scheduler.server_args.resolved_role_device = lambda: "cuda"
        scheduler._compute_stream = object()
        stream_state = {"active": False, "stream": None}
        current_stream = MagicMock()

        def _forward(batch):
            self.assertTrue(stream_state["active"])
            self.assertIs(stream_state["stream"], scheduler._compute_stream)
            return SimpleNamespace(
                output=torch.zeros(1),
                audio=None,
                audio_sample_rate=None,
                error=None,
            )

        scheduler.worker.execute_forward = MagicMock(side_effect=_forward)

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.torch.cuda.stream",
            side_effect=lambda stream: _TrackedStreamContext(stream_state, stream),
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.torch.cuda.current_stream",
            return_value=current_stream,
        ), patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.send_tensors"
        ) as mock_send_tensors:
            scheduler._disagg_decoder_compute(
                Req(request_id="dec-compute"),
                "dec-compute",
                "DECODER",
            )

        current_stream.wait_stream.assert_called_once_with(scheduler._compute_stream)
        mock_send_tensors.assert_called_once()

    def test_follower_compute_uses_compute_stream(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler.server_args.resolved_role_device = lambda: "cuda"
        scheduler._compute_stream = object()
        stream_state = {"active": False, "stream": None}

        def _forward(batch, return_req=False):
            self.assertTrue(stream_state["active"])
            self.assertIs(stream_state["stream"], scheduler._compute_stream)
            return batch[0]

        scheduler.worker.execute_forward = MagicMock(side_effect=_forward)

        with patch(
            "sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin.torch.cuda.stream",
            side_effect=lambda stream: _TrackedStreamContext(stream_state, stream),
        ):
            scheduler._disagg_compute_non_rank0({"request_id": "follower-compute"})

        scheduler.worker.execute_forward.assert_called_once()


class TestSchedulerWarmupCalibration(unittest.TestCase):
    def test_schedule_transfer_reconfigure_keeps_max_sizes(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler._transfer_manager = MagicMock()

        scheduler._schedule_transfer_reconfigure(128, 32)
        scheduler._schedule_transfer_reconfigure(64, 256)

        self.assertEqual(
            scheduler._pending_transfer_reconfigure,
            {"transfer_bytes": 128, "meta_bytes": 256},
        )

    def test_apply_pending_transfer_reconfigure_rebuilds_idle_manager(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        manager = MagicMock()
        manager.has_active_transfers.return_value = False
        scheduler._transfer_manager = manager
        scheduler._pending_transfer_reconfigure = {
            "transfer_bytes": 4096,
            "meta_bytes": 1024,
        }
        scheduler._preallocated_slots = {"stale": object()}
        scheduler._init_disagg_transfer_manager = MagicMock()

        rebuilt = scheduler._maybe_apply_pending_transfer_reconfigure()

        self.assertTrue(rebuilt)
        manager.cleanup.assert_called_once()
        scheduler._init_disagg_transfer_manager.assert_called_once_with(
            measured_transfer_bytes=4096,
            measured_meta_bytes=1024,
        )
        self.assertIsNone(scheduler._pending_transfer_reconfigure)
        self.assertTrue(scheduler._transfer_reconfigured)
        self.assertEqual(scheduler._preallocated_slots, {})

    def test_encoder_warmup_send_completion_schedules_reconfigure(self):
        scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        scheduler._transfer_manager = MagicMock()
        scheduler._transfer_manager.send_direct_message = MagicMock()
        scheduler._schedule_transfer_reconfigure = MagicMock()

        scheduler._on_direct_send_completion(
            "warmup-enc",
            SimpleNamespace(
                receiver_control_endpoint="tcp://receiver",
                prealloc_slot_id=3,
            ),
            SimpleNamespace(
                scalar_fields={"is_warmup": True},
                transfer_size=8192,
                meta_size=512,
            ),
            True,
            None,
        )

        scheduler._schedule_transfer_reconfigure.assert_called_once_with(8192, 512)

    def test_denoiser_warmup_send_completion_uses_max_inbound_and_outbound(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler._transfer_manager = MagicMock()
        scheduler._transfer_manager.send_direct_message = MagicMock()
        scheduler._schedule_transfer_reconfigure = MagicMock()
        scheduler._warmup_inbound_sizes["warmup-den"] = (2048, 1024)

        scheduler._on_direct_send_completion(
            "warmup-den",
            SimpleNamespace(
                receiver_control_endpoint="tcp://receiver",
                prealloc_slot_id=5,
            ),
            SimpleNamespace(
                scalar_fields={"is_warmup": True},
                transfer_size=4096,
                meta_size=256,
            ),
            True,
            None,
        )

        scheduler._schedule_transfer_reconfigure.assert_called_once_with(4096, 1024)
        self.assertNotIn("warmup-den", scheduler._warmup_inbound_sizes)

    def test_decoder_warmup_compute_schedules_reconfigure_from_inbound_sizes(self):
        scheduler = _SchedulerHarness.make(RoleType.DECODER)
        scheduler._release_pending_receive = MagicMock()
        scheduler._build_disagg_compute_req = MagicMock(return_value=Req(request_id="warmup-dec"))
        scheduler._disagg_decoder_compute = MagicMock()
        scheduler._schedule_transfer_reconfigure = MagicMock()
        scheduler._warmup_inbound_sizes["warmup-dec"] = (1536, 384)
        item = _PendingInboundTransfer(
            request_id="warmup-dec",
            role_name="DECODER",
            scalar_fields={"is_warmup": True},
            tensors={"latents": torch.randn(1, 4, 4, 4)},
            load_event=None,
            prealloc_slot_id=9,
        )

        scheduler._run_prefetched_compute_item(item, is_multi_rank=False)

        scheduler._schedule_transfer_reconfigure.assert_called_once_with(1536, 384)
        self.assertNotIn("warmup-dec", scheduler._warmup_inbound_sizes)


class TestSchedulerTensorDistribution(unittest.TestCase):
    def test_denoiser_stage_managed_sp_shards_and_marks_req(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler.server_args.sp_degree = 2
        scheduler.server_args.pipeline_config = SimpleNamespace(
            shard_latents_for_sp=MagicMock(
                side_effect=lambda req, tensor: (tensor[..., :1].clone(), True)
            )
        )
        scheduler._broadcast_tensor_payload_to_all_ranks = MagicMock(
            side_effect=lambda payload: payload
        )

        latents = torch.randn(1, 2, 4, 4)
        image_latent = torch.randn(1, 2, 4, 4)
        prompt_embeds = torch.randn(1, 8, 16)

        req = scheduler._build_disagg_compute_req(
            {"request_id": "req-sp-1", "enable_sequence_shard": False},
            {
                "latents": latents,
                "image_latent": image_latent,
                "prompt_embeds": prompt_embeds,
            },
        )

        self.assertEqual(tuple(req.latents.shape), (1, 2, 4, 1))
        self.assertEqual(tuple(req.image_latent.shape), (1, 2, 4, 1))
        self.assertEqual(tuple(req.prompt_embeds.shape), tuple(prompt_embeds.shape))
        self.assertEqual(
            getattr(req, "_disagg_pre_sharded_fields"),
            ("image_latent", "latents"),
        )
        self.assertEqual(
            scheduler.server_args.pipeline_config.shard_latents_for_sp.call_count, 2
        )

    def test_denoiser_model_managed_sp_keeps_full_inputs(self):
        scheduler = _SchedulerHarness.make(RoleType.DENOISER)
        scheduler.server_args.sp_degree = 2
        scheduler.server_args.pipeline_config = SimpleNamespace(
            shard_latents_for_sp=MagicMock()
        )
        scheduler._broadcast_tensor_payload_to_all_ranks = MagicMock(
            side_effect=lambda payload: payload
        )

        latents = torch.randn(1, 2, 4, 4)
        req = scheduler._build_disagg_compute_req(
            {"request_id": "req-sp-2", "enable_sequence_shard": True},
            {"latents": latents},
        )

        self.assertTrue(torch.equal(req.latents, latents))
        self.assertFalse(hasattr(req, "_disagg_pre_sharded_fields"))
        scheduler.server_args.pipeline_config.shard_latents_for_sp.assert_not_called()

    def test_decoder_parallel_decode_keeps_full_latents(self):
        scheduler = _SchedulerHarness.make(RoleType.DECODER)
        scheduler.server_args.sp_degree = 2
        scheduler._broadcast_tensor_payload_to_all_ranks = MagicMock(
            side_effect=lambda payload: payload
        )
        scheduler.worker.pipeline = SimpleNamespace(
            get_module=lambda name: SimpleNamespace(use_parallel_decode=True)
            if name == "vae"
            else None
        )

        latents = torch.randn(1, 4, 4, 4, 4)
        req = scheduler._build_disagg_compute_req(
            {"request_id": "req-dec-1"},
            {"latents": latents},
        )

        self.assertTrue(torch.equal(req.latents, latents))
        self.assertFalse(hasattr(req, "_disagg_pre_sharded_fields"))


class TestDenoisingStagePreShardedInputs(unittest.TestCase):
    def test_preprocess_sp_latents_skips_disagg_pre_sharded_fields(self):
        batch = Req(request_id="req-stage-1")
        batch.latents = torch.randn(1, 4, 4, 4, 4)
        batch.image_latent = torch.randn(1, 4, 4, 4, 4)
        batch._disagg_pre_sharded_fields = ("latents", "image_latent")

        pipeline_config = SimpleNamespace(shard_latents_for_sp=MagicMock())
        stage = object.__new__(DenoisingStage)

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.get_sp_world_size",
            return_value=2,
        ):
            DenoisingStage._preprocess_sp_latents(
                stage,
                batch,
                SimpleNamespace(pipeline_config=pipeline_config),
            )

        self.assertTrue(batch.did_sp_shard_latents)
        pipeline_config.shard_latents_for_sp.assert_not_called()


class TestSchedulerTransferEncoderStaging(unittest.TestCase):
    def setUp(self):
        MockTransferEngine.reset()
        self.addCleanup(MockTransferEngine.reset)
        self.engine = MockTransferEngine(session_id="encoder-session")
        self.buffer = TransferTensorBuffer(pool_size=2 * 1024 * 1024, role_name="test")
        self.meta_buffer = TransferMetaBuffer(slot_count=2, slot_size=64 * 1024, role_name="test")
        self.tm = DiffusionTransferManager(
            engine=self.engine,
            buffer=self.buffer,
            meta_buffer=self.meta_buffer,
            host_id="host-a",
        )
        self.addCleanup(self.tm.cleanup)
        self.scheduler = _SchedulerHarness.make(RoleType.ENCODER)
        self.scheduler._transfer_manager = self.tm

    def test_encoder_transfer_stage_enqueues_then_sends_staged_msg(self):
        tensor_fields = {
            "prompt_embeds": torch.randn(1, 8, 32),
            "latents": torch.randn(1, 4, 16, 16),
        }
        scalar_fields = {"request_id": "req-enc-1", "guidance_scale": 7.5}

        self.scheduler._disagg_encoder_transfer_stage(
            "req-enc-1", tensor_fields, scalar_fields
        )

        self.assertEqual(len(self.scheduler._swap_out_queue), 1)
        self.assertTrue(self.scheduler._process_swap_out_queue_once())
        self.assertTrue(self.scheduler._process_send_ready_queue_once())

        self.scheduler._pool_result_push.send_multipart.assert_called_once()
        sent_frames = self.scheduler._pool_result_push.send_multipart.call_args[0][0]
        staged_msg = decode_transfer_msg(sent_frames)
        self.assertEqual(staged_msg["msg_type"], TransferMsgType.STAGED)
        self.assertEqual(staged_msg["request_id"], "req-enc-1")
        self.assertEqual(staged_msg["session_id"], self.engine.session_id)
        self.assertGreater(staged_msg["meta_size"], 0)


if __name__ == "__main__":
    unittest.main()
