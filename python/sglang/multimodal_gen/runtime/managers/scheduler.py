# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import pickle
import time
from collections import deque
from typing import Any, List

import zmq

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    SchedulerDisaggMixin,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import send_tensors
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TransferCreditMsg,
    encode_transfer_msg,
)
from sglang.multimodal_gen.runtime.ddit.config import (
    build_execution_plan,
    resolve_resolution_key,
    resolve_schedule_policy,
    resolve_vae_ranks,
)
from sglang.multimodal_gen.runtime.ddit.concurrent import (
    CommandWave,
    CommandWaveBuilder,
    DDiTOp,
)
from sglang.multimodal_gen.runtime.ddit.dynamic_sp import (
    activation_rank_tuples_for_server,
    get_dynamic_sp_registry,
    prebuild_rank_tuples_for_server,
)
from sglang.multimodal_gen.runtime.ddit.logging import record_lifecycle
from sglang.multimodal_gen.runtime.ddit.logging import record_op_trace_rows
from sglang.multimodal_gen.runtime.ddit.logging import record_rank_switch
from sglang.multimodal_gen.runtime.ddit.scheduler import (
    DDiTRequestState,
    FixedBaselineScheduler,
    ForcedSwitchScheduler,
    HungryFirstScheduler,
    RequestPhase,
    build_fixed_baseline_scheduler_config,
    build_forced_switch_scheduler_config,
    build_hungry_scheduler_config,
    build_profile_scheduler,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    GetWeightsChecksumReqInput,
    UpdateWeightFromDiskReqInput,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    GetDisaggStatsReq,
    ListLorasReq,
    MergeLoraWeightsReq,
    SetLoraReq,
    ShutdownReq,
    UnmergeLoraWeightsReq,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_group
from sglang.multimodal_gen.runtime.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args,
)
from sglang.multimodal_gen.runtime.warmup_utils import (
    build_server_warmup_reqs as build_warmup_reqs,
)
from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket
from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj
from sglang.multimodal_gen.runtime.utils.logging_utils import GREEN, RESET, init_logger
from sglang.multimodal_gen.runtime.utils.request_profiling import (
    RequestCsvProfiler,
    flatten_request_metrics,
    resolve_profile_dir,
)

logger = init_logger(__name__)

class Scheduler(SchedulerDisaggMixin):
    """
    Runs the main event loop for the rank 0 worker.
    It listens for external requests via ZMQ and coordinates with other workers.
    This class does NOT manage worker processes.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        port_args: PortArgs,
        task_pipes_to_slaves: list = None,
        result_pipes_from_slaves: list = None,
        local_rank: int | None = None,
    ):
        self.server_args = server_args
        self.port_args = port_args

        # local_rank is the physical GPU index for torch.cuda.set_device.
        # In non-disagg mode, it equals gpu_id. In disagg mode, it may differ
        # (e.g., denoiser rank 0 on physical GPU 1).
        if local_rank is None:
            local_rank = gpu_id

        set_global_server_args(server_args=server_args)

        # Inter-process Communication
        self.context = zmq.Context(io_threads=2)
        endpoint = server_args.scheduler_endpoint
        if gpu_id == 0:
            # router allocates identify (envelope) for each connection
            self.receiver, actual_endpoint = get_zmq_socket(
                self.context, zmq.ROUTER, endpoint, True
            )
            logger.info(f"Scheduler bind at endpoint: {actual_endpoint}")
        else:
            self.receiver = None

        worker = GPUWorker(
            local_rank=local_rank,
            master_port=port_args.master_port,
            rank=gpu_id,
            server_args=server_args,
        )
        self.worker = worker
        self.task_pipes_to_slaves = task_pipes_to_slaves
        self.result_pipes_from_slaves = result_pipes_from_slaves
        self.gpu_id = gpu_id
        self._running = True

        self.request_handlers = {
            SetLoraReq: self._handle_set_lora,
            MergeLoraWeightsReq: self._handle_merge_lora,
            UnmergeLoraWeightsReq: self._handle_unmerge_lora,
            Req: self._handle_generation,
            List[Req]: self._handle_generation,
            ListLorasReq: self._handle_list_loras,
            ShutdownReq: self._handle_shutdown,
            GetDisaggStatsReq: self._handle_get_disagg_stats,
            UpdateWeightFromDiskReqInput: self._handle_update_weights_from_disk,
            GetWeightsChecksumReqInput: self._handle_get_weights_checksum,
        }

        # FIFO, new reqs are appended
        self.waiting_queue: deque[tuple[bytes, Req]] = deque()

        # whether we've send the necessary warmup reqs
        self.warmed_up = False
        # warmup progress tracking
        self._warmup_total = 0
        self._warmup_processed = 0

        # Maximum consecutive errors before terminating the event loop
        self._max_consecutive_errors = 3
        self._consecutive_error_count = 0

        self._init_disagg_state(server_args, local_rank)
        self._monolithic_profile_writer = None
        if (
            self.gpu_id == 0
            and self._disagg_role == RoleType.MONOLITHIC
            and getattr(server_args, "profile_enabled", False)
        ):
            profile_dir = resolve_profile_dir(
                server_args.profile_output_dir,
                server_args.profile_run_id,
                deployment_mode="monolithic",
            )
            self._monolithic_profile_writer = RequestCsvProfiler(
                f"{profile_dir}/monolithic_server.csv"
            )
        if self._disagg_role != RoleType.MONOLITHIC:
            self._run_disagg_startup_warmup(self.build_server_warmup_reqs())
            self.warmed_up = True
        else:
            self.prepare_server_warmup_reqs()

    def get_disagg_metrics(self) -> dict | None:
        """Return disagg role metrics snapshot, or None if not in disagg mode."""
        if self._disagg_metrics is None:
            return None
        return self._disagg_metrics.snapshot().to_dict()

    def _handle_get_disagg_stats(self, _reqs: List[Any]) -> OutputBatch:
        """Handle stats request — return disagg metrics via OutputBatch.output."""
        stats = self.get_disagg_metrics()
        return OutputBatch(
            output=stats or {"role": "monolithic", "message": "not in disagg mode"}
        )

    def _handle_set_lora(self, reqs: List[Any]) -> OutputBatch:
        # TODO: return set status
        # TODO: return with SetLoRAResponse or something more appropriate
        req = reqs[0]
        return self.worker.set_lora(
            req.lora_nickname, req.lora_path, req.target, req.strength
        )

    def _handle_merge_lora(self, reqs: List[Any]):
        req = reqs[0]
        return self.worker.merge_lora_weights(req.target, req.strength)

    def _handle_unmerge_lora(self, reqs: List[Any]) -> OutputBatch:
        req = reqs[0]
        return self.worker.unmerge_lora_weights(req.target)

    def _handle_list_loras(self, _reqs: List[Any]) -> OutputBatch:
        return self.worker.list_loras()

    def _handle_shutdown(self, _reqs: List[Any]) -> OutputBatch:
        self._running = False
        return OutputBatch()

    def _handle_update_weights_from_disk(self, reqs: List[Any]) -> OutputBatch:
        """Handle update_weights_from_disk request for RL workflows."""
        req = reqs[0]
        success, message = self.worker.update_weights_from_disk(
            model_path=req.model_path,
            flush_cache=req.flush_cache,
            target_modules=req.target_modules,
        )
        return OutputBatch(
            output={"success": success, "message": message},
            error=None if success else message,
        )

    def _handle_get_weights_checksum(self, reqs: List[Any]) -> OutputBatch:
        """Handle get_weights_checksum request."""
        req = reqs[0]
        checksums = self.worker.get_weights_checksum(module_names=req.module_names)
        return OutputBatch(output=checksums)

    def _handle_generation(self, reqs: List[Req]):
        warmup_reqs = [req for req in reqs if req.is_warmup]
        if warmup_reqs:
            self._warmup_processed += len(warmup_reqs)
            if self._warmup_total > 0:
                logger.info(
                    f"Processing warmup req... ({self._warmup_processed}/{self._warmup_total})"
                )
            else:
                logger.info("Processing warmup req...")

        return self.worker.execute_forward(reqs)

    def return_result(
        self,
        output_batch: OutputBatch,
        identity: bytes | None = None,
        is_warmup: bool = False,
    ):
        """
        replies to client, only on rank 0
        """
        if not is_warmup and self.receiver is not None and identity is not None:
            self.receiver.send_multipart([identity, b"", pickle.dumps(output_batch)])

    def get_next_batch_to_run(self) -> list[tuple[bytes, Req]] | None:
        """pull a req from waiting_queue"""
        if not self.waiting_queue:
            return None

        # pop the first (earliest)
        item = self.waiting_queue.popleft()
        req = item[1]
        if (
            isinstance(req, Req)
            and not req.is_warmup
            and req.metrics is not None
            and req.metrics.start_time_s is None
        ):
            req.metrics.start_time_s = time.time()

        return [item]

    def _write_monolithic_profile_row(
        self,
        req: Req | Any,
        output_batch: OutputBatch,
    ) -> None:
        if (
            self._monolithic_profile_writer is None
            or not isinstance(req, Req)
            or req.is_warmup
            or (req.metrics is None and output_batch.metrics is None)
        ):
            return

        metrics = output_batch.metrics or req.metrics
        if metrics.finish_time_s is None:
            metrics.finish_time_s = time.time()

        row = flatten_request_metrics(metrics)
        row["status"] = "failed" if output_batch.error else "completed"
        row["error"] = output_batch.error
        if output_batch.output_file_paths:
            row["output_file_paths"] = output_batch.output_file_paths
        self._monolithic_profile_writer.write_row(row)

    def build_server_warmup_reqs(self) -> list[Req]:
        if self.warmed_up or not self.server_args.warmup:
            return []

        warmup_reqs = build_warmup_reqs(self.server_args)
        self._warmup_total = len(warmup_reqs)
        self._warmup_processed = 0
        return warmup_reqs

    def prepare_server_warmup_reqs(self):
        warmup_reqs = self.build_server_warmup_reqs()
        for req in warmup_reqs:
            self.waiting_queue.append((None, req))
        if warmup_reqs:
            # if server is warmed-up, set this flag to avoid req-based warmup
            self.warmed_up = True

    def process_received_reqs_with_req_based_warmup(
        self, recv_reqs: List[tuple[bytes, Any]]
    ) -> List[tuple[bytes, Any]]:
        if (
            self.warmed_up
            or not self.server_args.warmup
            or not recv_reqs
            or self.server_args.warmup_resolutions is not None
        ):
            return recv_reqs

        # handle server req-based warmup by inserting an identical req to the beginning of the waiting queue
        # only the very first req through server's lifetime will be warmed up
        identity, req = recv_reqs[0]
        if isinstance(req, Req):
            warmup_req = req.copy_as_warmup(self.server_args.warmup_steps)
            recv_reqs.insert(0, (identity, warmup_req))
            self._warmup_total = 1
            self._warmup_processed = 0
            self.warmed_up = True
        return recv_reqs

    def recv_reqs(self) -> List[tuple[bytes, Any]]:
        """
        For non-main schedulers, reqs are broadcasted from main using broadcast_pyobj
        """
        if self.receiver is not None:
            try:
                try:
                    # Accept valid REQ envelopes only, ignore malformed/probe frames.
                    parts = self.receiver.recv_multipart(zmq.NOBLOCK)
                    identity, payload = parts[0], parts[-1]

                    # Ignore malformed probes or non-pickle data
                    recv_reqs = pickle.loads(payload) if len(parts) > 2 else []
                except (zmq.Again, pickle.UnpicklingError, IndexError, EOFError):
                    recv_reqs = []
            except zmq.ZMQError:
                # re-raise or handle appropriately to let the outer loop continue
                raise

            if recv_reqs:
                # Ensure recv_reqs is a list
                if not isinstance(recv_reqs, list):
                    recv_reqs = [recv_reqs]

                # Pack with identity for rank 0
                recv_reqs = [(identity, req) for req in recv_reqs]
        else:
            recv_reqs = None

        # TODO: fix this condition
        if self.server_args.sp_degree != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.sp_group.rank,
                self.worker.sp_cpu_group,
                src=self.worker.sp_group.ranks[0],
            )

        if self.server_args.enable_cfg_parallel:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.cfg_group.rank,
                self.worker.cfg_cpu_group,
                src=self.worker.cfg_group.ranks[0],
            )

        if self.server_args.tp_size > 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.tp_group.rank,
                self.worker.tp_cpu_group,
                src=self.worker.tp_group.ranks[0],
            )

        assert recv_reqs is not None

        return recv_reqs

    def event_loop(self) -> None:
        """
        The main event loop that listens for ZMQ requests.
        Handles abortion
        """
        # Pool mode: DDiT workers use the concurrent DiT/VAE runtime after
        # receiving encoder-prepared requests through the disagg transfer path.
        if self._disagg_role == RoleType.DDIT_WORKER:
            self._ddit_concurrent_event_loop(
                resolve_schedule_policy(self.server_args),
                disagg_prepared=True,
            )
            return
        if self._disagg_role != RoleType.MONOLITHIC:
            self._disagg_event_loop()
            return
        schedule_policy = resolve_schedule_policy(self.server_args)
        if (
            getattr(self.server_args, "enable_ddit", False)
            and schedule_policy
            in (
                "hungry_first",
                "fixed_baseline",
                "naive",
                "naive_greedy",
                "wsjf",
                "wsjf_scale_up",
            )
        ):
            self._ddit_concurrent_event_loop(schedule_policy)
            return

        logger.debug(
            f"Rank 0 scheduler listening on tcp://*:{self.server_args.scheduler_port}"
        )

        while self._running:
            # Update queue depth for metrics
            if self._disagg_metrics:
                self._disagg_metrics.update_queue_depth(len(self.waiting_queue))

            # 1: receive requests
            try:
                new_reqs = self.recv_reqs()
                new_reqs = self.process_received_reqs_with_req_based_warmup(new_reqs)
                now_s = time.time()
                for _, req in new_reqs:
                    if (
                        isinstance(req, Req)
                        and not req.is_warmup
                        and req.metrics is not None
                        and req.metrics.arrival_time_s is None
                    ):
                        req.metrics.arrival_time_s = now_s
                    if isinstance(req, Req) and not req.is_warmup:
                        record_lifecycle(
                            self.server_args,
                            req,
                            "add",
                            timestamp=now_s,
                            status="queued",
                        )
                self.waiting_queue.extend(new_reqs)
                # Reset error count on success
                self._consecutive_error_count = 0
            except Exception as e:
                self._consecutive_error_count += 1
                logger.error(
                    f"Error receiving requests in scheduler event loop "
                    f"(attempt {self._consecutive_error_count}/{self._max_consecutive_errors}): {e}",
                    exc_info=True,
                )
                if self._consecutive_error_count >= self._max_consecutive_errors:
                    logger.error(
                        f"Maximum consecutive errors ({self._max_consecutive_errors}) reached. "
                        "Terminating scheduler event loop."
                    )
                    raise RuntimeError(
                        f"Scheduler terminated after {self._max_consecutive_errors} "
                        f"consecutive errors. Last error: {e}"
                    ) from e
                continue

            # 2: execute, make sure a reply is always sent
            items = self.get_next_batch_to_run()
            if not items:
                continue

            identities = [item[0] for item in items]
            reqs = [item[1] for item in items]

            try:
                processed_req = reqs[0]
                is_warmup = (
                    processed_req.is_warmup if isinstance(processed_req, Req) else False
                )

                handler = self.request_handlers.get(type(processed_req))
                if handler:
                    output_batch = handler(reqs)
                else:
                    output_batch = OutputBatch(
                        error=f"Unknown request type: {type(processed_req)}"
                    )
            except Exception as e:
                logger.error(
                    f"Error executing request in scheduler event loop: {e}",
                    exc_info=True,
                )
                output_batch = OutputBatch(error=str(e))
                if isinstance(processed_req, Req) and processed_req.metrics is not None:
                    output_batch.metrics = processed_req.metrics

            # 3. return results
            try:
                is_warmup = (
                    processed_req.is_warmup if isinstance(processed_req, Req) else False
                )
                if is_warmup:
                    if output_batch.error is None:
                        if self._warmup_total > 0:
                            logger.info(
                                f"Warmup req ({self._warmup_processed}/{self._warmup_total}) processed in {GREEN}%.2f{RESET} seconds",
                                output_batch.metrics.total_duration_s,
                            )
                        else:
                            logger.info(
                                f"Warmup req processed in {GREEN}%.2f{RESET} seconds",
                                output_batch.metrics.total_duration_s,
                            )
                    else:
                        if self._warmup_total > 0:
                            logger.info(
                                f"Warmup req ({self._warmup_processed}/{self._warmup_total}) processing failed"
                            )
                        else:
                            logger.info(f"Warmup req processing failed")

                self._write_monolithic_profile_row(processed_req, output_batch)
                # TODO: Support sending back to multiple identities if batched
                self.return_result(output_batch, identities[0], is_warmup=is_warmup)
            except zmq.ZMQError as e:
                # Reply failed; log and keep loop alive to accept future requests
                logger.error(f"ZMQ error sending reply: {e}")
                continue

        if self.receiver is not None:
            self.receiver.close()
        self._cleanup_disagg()
        self.context.destroy(linger=0)

    def _broadcast_task(self, payload: dict[str, Any]) -> None:
        """Broadcast a task to all slave worker processes."""
        method = payload["method"]
        kwargs = {k: v for k, v in payload.items() if k != "method"}
        task = {"method": method, "kwargs": kwargs}
        for pipe in self.task_pipes_to_slaves:
            pipe.send(task)

    def _collect_slave_results(self) -> List[dict[str, Any]]:
        """Collect results from all slave worker processes."""
        results = []
        for pipe in self.result_pipes_from_slaves:
            results.append(pipe.recv())
        return results

    def _ddit_run_rank_command(self, command: dict[str, Any]) -> Any:
        action = command["action"]
        if action == "idle":
            sleep_s = float(command.get("sleep_s", 0.0))
            if sleep_s > 0:
                time.sleep(sleep_s)
            return {"status": "idle"}
        if action == "full_prepare":
            return self.worker.prepare_hungry_request(command["req"])
        if action == "register_prepared":
            req = self._build_disagg_compute_req(
                command["scalar_fields"],
                command.get("tensors"),
            )
            disagg_role = getattr(self.server_args, "disagg_role", "")
            if (
                disagg_role == RoleType.DDIT_WORKER
                or str(disagg_role) == RoleType.DDIT_WORKER.value
            ):
                # The original frontend request owns output persistence.  The
                # ddit_worker only returns decoded tensors to DiffusionServer.
                req.save_output = False
                req.return_file_paths_only = False
            result = self.worker.register_hungry_prepared_request(req)
            if self.gpu_id == 0:
                result["req"] = req
            return result
        if action == "ensure_dynamic_sp":
            target_ranks = tuple(command.get("target_ranks") or command["ranks"])
            result = get_dynamic_sp_registry(self.server_args).ensure(target_ranks)
            if self.gpu_id == min(tuple(command["ranks"])):
                logger.info(
                    "DDiT worker: ensured dynamic SP group for ranks=%s "
                    "(cache_hit=%s, created=%s, degree_pair=%sx%s, reason=%s, "
                    "created_process_groups=%s, reused_process_groups=%s, "
                    "build_ms=%.2f, new_group_ms=%.2f, collective_touch_ms=%.2f)",
                    result.spec.ranks,
                    not result.created,
                    result.created,
                    result.spec.ulysses_degree,
                    result.spec.ring_degree,
                    command.get("log_reason", ""),
                    result.stats.created_process_groups,
                    result.stats.reused_process_groups,
                    result.stats.build_ms,
                    result.stats.new_group_ms,
                    result.stats.collective_touch_ms,
                )
            return {
                "target_ranks": result.spec.ranks,
                "created": result.created,
                "ulysses_degree": result.spec.ulysses_degree,
                "ring_degree": result.spec.ring_degree,
                "created_process_groups": result.stats.created_process_groups,
                "reused_process_groups": result.stats.reused_process_groups,
                "build_ms": result.stats.build_ms,
                "pg_build_ms": result.stats.build_ms,
                "new_group_ms": result.stats.new_group_ms,
                "collective_touch_ms": result.stats.collective_touch_ms,
                "prebuild_mode": str(
                    getattr(self.server_args, "ddit_dynamic_sp_prebuild_mode", "auto")
                    or "auto"
                ),
            }
        if action == "activate_dynamic_sp":
            started = time.perf_counter()
            result = self.worker.activate_hungry_dit(
                command["request_id"],
                tuple(command["ranks"]),
                tuple(command.get("activation_key") or ()),
                begin_index=int(command.get("begin_index") or 0),
                force=bool(command.get("force_activation", False)),
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            activation_ms = float(result.get("activation_ms", elapsed_ms))
            if self.gpu_id == min(tuple(command["ranks"])):
                logger.info(
                    "DDiT worker: activated dynamic SP group for ranks=%s "
                    "(cache_hit=%s, activation_ms=%.2f, reason=%s)",
                    tuple(command["ranks"]),
                    bool(result.get("cache_hit")),
                    activation_ms,
                    command.get("log_reason", ""),
                )
            timings = result.get("timings") or {}
            return {
                "request_id": command["request_id"],
                "target_ranks": tuple(command["ranks"]),
                "activation_key": tuple(command.get("activation_key") or ()),
                "activation_cache_hit": bool(result.get("cache_hit")),
                "activation_ms": activation_ms,
                "collective_warmup_ms": 0.0,
                "model_forward_warmup_ms": float(
                    timings.get("predict_noise_ms", 0.0)
                ),
                "activation_timings": timings,
                "activation_key_scope": "coarse",
                "force_activation": bool(command.get("force_activation", False)),
                "prebuild_mode": str(
                    getattr(self.server_args, "ddit_dynamic_sp_prebuild_mode", "auto")
                    or "auto"
                ),
            }
        if action == "warmup_ack":
            self.worker.drop_hungry_request(command["request_id"])
            if self.gpu_id == 0:
                self._ddit_send_output_to_disagg_server(
                    command["request_id"], OutputBatch()
                )
            return {"request_id": command["request_id"], "status": "warmup_ack"}
        if action == "full_forward":
            return self.worker.execute_forward([command["req"]])
        if action == "control":
            req = command["req"]
            handler = self.request_handlers.get(type(req))
            if handler is None:
                return OutputBatch(error=f"Unknown request type: {type(req)}")
            return handler([req])
        if action == "dit_init":
            return self.worker.start_hungry_dit(
                command["request_id"],
                tuple(command["ranks"]),
                log_events=False,
                log_stage=command.get("log_stage", "dit"),
                log_reason=command.get("log_reason", "waiting_queue"),
                log_policy=command.get("log_policy", "hungry_first"),
            )
        if action == "dit_step":
            return self.worker.run_hungry_dit_step(
                command["request_id"], tuple(command["ranks"]), log_events=False
            )
        if action == "dit_migrate":
            return self.worker.migrate_hungry_dit(
                command["request_id"],
                tuple(command["old_ranks"]),
                tuple(command["new_ranks"]),
                int(command["step"]),
            )
        if action == "dit_finish":
            return self.worker.finish_hungry_dit(
                command["request_id"],
                log_events=False,
                broadcast_world=False,
            )
        if action == "vae_prepare":
            return self.worker.prepare_hungry_vae(
                command["request_id"],
                tuple(command["final_dit_ranks"]),
                tuple(command["vae_ranks"]),
            )
        if action == "vae_run":
            return self.worker.run_concurrent_vae(
                command["request_id"], tuple(command["ranks"])
            )
        if action == "output_transfer":
            return self.worker.transfer_concurrent_output(
                command["request_id"], int(command["src_rank"])
            )
        if action == "shutdown":
            self._running = False
            return OutputBatch()
        raise ValueError(f"Unknown concurrent DDiT action: {action}")

    def _ddit_execute_rank_command(self, command: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        try:
            result = self._ddit_run_rank_command(command)
            status = "ok"
            error = ""
        except Exception as e:
            logger.error("Concurrent DDiT rank command failed: %s", e, exc_info=True)
            result = None
            status = "failed"
            error = str(e)
        ended = time.time()
        trace = {
            "timestamp_start": started,
            "timestamp_end": ended,
            "request_id": command.get("request_id"),
            "stage": command.get("stage", command.get("action")),
            "step": command.get("step"),
            "ranks": list(command.get("ranks", ())),
            "wave_id": command.get("wave_id"),
            "op_id": command.get("op_id"),
            "rank": self.gpu_id,
            "action": command.get("action"),
            "status": status,
            "error": error,
        }
        if isinstance(result, dict):
            for key in (
                "activation_ms",
                "activation_cache_hit",
                "model_forward_warmup_ms",
                "collective_warmup_ms",
                "collective_touch_ms",
                "pg_build_ms",
                "build_ms",
                "new_group_ms",
                "activation_key_scope",
                "force_activation",
                "prebuild_mode",
                "activation_timings",
                "created_process_groups",
                "reused_process_groups",
            ):
                if key in result:
                    trace[key] = result[key]
        return {
            "rank": self.gpu_id,
            "status": status,
            "error": error,
            "command": command,
            "result": result,
            "trace": trace,
        }

    def _ddit_slave_concurrent_loop(self) -> None:
        task_pipe = self.task_pipes_to_slaves
        result_pipe = self.result_pipes_from_slaves
        while self._running:
            command = task_pipe.recv()
            result_pipe.send(self._ddit_execute_rank_command(command))

    def _ddit_run_wave(self, wave: CommandWave, world_size: int) -> list[dict[str, Any]]:
        commands = wave.commands_by_rank(world_size)
        for rank in range(1, world_size):
            command = commands[rank]
            if command.get("action") == "register_prepared":
                command = dict(command)
                command.pop("tensors", None)
            self.task_pipes_to_slaves[rank - 1].send(command)
        results = [self._ddit_execute_rank_command(commands[0])]
        for rank in range(1, world_size):
            results.append(self.result_pipes_from_slaves[rank - 1].recv())
        non_idle_results = [
            result
            for result in results
            if result["command"].get("action") != "idle"
        ]
        warmup_only_wave = bool(non_idle_results) and all(
            str(result["command"].get("request_id") or "").startswith("warmup-")
            for result in non_idle_results
        )
        trace_rows = [
            result["trace"]
            for result in non_idle_results
            if not str(result["trace"].get("request_id") or "").startswith("warmup-")
        ]
        if trace_rows and not warmup_only_wave:
            record_op_trace_rows(self.server_args, trace_rows)
        failed = [result for result in results if result["status"] != "ok"]
        if failed:
            raise RuntimeError(failed[0]["error"])
        return results

    def _ddit_result_for_op(
        self, results: list[dict[str, Any]], op: DDiTOp
    ) -> Any:
        ranks = set(op.ranks)
        for result in sorted(results, key=lambda item: item["rank"]):
            if result["rank"] in ranks and result["command"].get("op_id") == (
                op.op_id or result["command"].get("op_id")
            ):
                return result["result"]
        return None

    def _ddit_exclusive_wave(
        self,
        *,
        wave_id: int,
        action: str,
        ranks: tuple[int, ...],
        payload: dict[str, Any],
        request_id: str | None = None,
        stage: str | None = None,
    ) -> CommandWave:
        return CommandWave(
            wave_id,
            (
                DDiTOp(
                    action=action,
                    request_id=request_id,
                    ranks=ranks,
                    stage=stage or action,
                    payload=payload,
                    op_id=f"wave{wave_id}:{action}",
                ),
            ),
        )

    def _ddit_policy_requests(self, policy: Any) -> dict[str, DDiTRequestState]:
        return policy.requests

    def _ddit_policy_complete(self, policy: Any, request_id: str) -> None:
        if getattr(policy, "vae_same_as_dit", False):
            policy.complete_request(request_id)
        elif isinstance(policy, HungryFirstScheduler):
            policy.complete_vae(request_id)
        else:
            policy.complete_request(request_id)

    def _ddit_policy_update_step(
        self, policy: Any, request_id: str, cur_step: int
    ) -> None:
        if hasattr(policy, "update_cur_step"):
            policy.update_cur_step(request_id, cur_step)
        else:
            policy.requests[request_id].cur_step = int(cur_step)

    def _ddit_policy_fail_request(self, policy: Any, request_id: str) -> None:
        state = getattr(policy, "requests", {}).get(request_id)
        if state is None:
            return
        gpu_owner = getattr(policy, "gpu_owner", None)
        if isinstance(gpu_owner, dict):
            for rank, owner in list(gpu_owner.items()):
                if owner == request_id:
                    gpu_owner[rank] = None
        for attr in ("waiting", "dit_waiting", "text_encoder_queue", "window"):
            queue = getattr(policy, attr, None)
            if isinstance(queue, deque):
                self._ddit_remove_request_ids_from_deque(queue, {request_id})
        state.ranks = ()
        state.phase = RequestPhase.DONE

    def _ddit_remove_request_ids_from_deque(
        self, queue: deque[Any], request_ids: set[str]
    ) -> None:
        retained: deque[Any] = deque()
        while queue:
            item = queue.popleft()
            item_request_id = (
                item.request_id if isinstance(item, DDiTOp) else str(item)
            )
            if item_request_id not in request_ids:
                retained.append(item)
        queue.extend(retained)

    def _ddit_fail_concurrent_wave_requests(
        self,
        *,
        wave: CommandWave,
        error: str,
        policy: Any,
        running_order: deque[str],
        pending_queues: list[deque[DDiTOp]],
        tracking_sets: list[set[str]],
        ensuring_dynamic_sp: set[tuple[int, ...]],
        activating_dynamic_sp: set[tuple[Any, ...]],
        identities: dict[str, bytes | None],
        req_by_id: dict[str, Req],
        disagg_prepared: bool,
    ) -> None:
        request_ids = {
            str(op.request_id)
            for op in wave.ops
            if op.request_id and not str(op.request_id).startswith("warmup-")
        }
        for op in wave.ops:
            if op.action == "ensure_dynamic_sp":
                target_ranks = tuple(op.payload.get("target_ranks") or op.ranks)
                ensuring_dynamic_sp.discard(target_ranks)
            elif op.action == "activate_dynamic_sp":
                activation_key = tuple(op.payload.get("activation_key") or ())
                activating_dynamic_sp.discard(activation_key)
        if not request_ids:
            return

        self._ddit_remove_request_ids_from_deque(running_order, request_ids)
        for queue in pending_queues:
            self._ddit_remove_request_ids_from_deque(queue, request_ids)
        for tracking_set in tracking_sets:
            tracking_set.difference_update(request_ids)

        for request_id in sorted(request_ids):
            req = req_by_id.get(request_id)
            if req is not None and not self._ddit_is_warmup_req(req):
                record_lifecycle(self.server_args, req, "vae_end", error=error)
            self._ddit_policy_fail_request(policy, request_id)

            if self.gpu_id == 0:
                output = OutputBatch(error=error)
                if disagg_prepared:
                    self._ddit_send_output_to_disagg_server(request_id, output)
                else:
                    identity = identities.get(request_id)
                    if identity is not None:
                        self.return_result(output, identity, is_warmup=False)
            identities.pop(request_id, None)
            req_by_id.pop(request_id, None)

    def _ddit_log_registered_request_state(
        self,
        *,
        source: str,
        req: Req,
        state: DDiTRequestState,
        schedule_policy: str,
    ) -> None:
        switch_plan = [
            {
                "after_step": int(event.after_step),
                "ranks": list(event.ranks),
                "reason": event.reason,
            }
            for event in state.switch_plan
        ]
        extra = getattr(req, "extra", {}) or {}
        logger.info(
            "DDiT worker: registered %s request %s "
            "(resolution=%s, steps=%d, policy=%s, initial_ranks=%s, "
            "switch_plan=%s, vae_k=%s, vae_ranks=%s)",
            source,
            state.request_id,
            state.resolution,
            state.total_steps,
            schedule_policy,
            state.initial_ranks,
            switch_plan,
            extra.get("ddit_vae_k"),
            extra.get("ddit_vae_ranks"),
        )
        if schedule_policy == "forced_switch" and not state.switch_plan:
            logger.warning(
                "DDiT worker: forced_switch request %s has empty switch_plan "
                "(req_extra_keys=%s, req_extra_ddit_switch_plan=%r, "
                "server_ddit_switch_plan=%r)",
                state.request_id,
                sorted(extra.keys()),
                extra.get("ddit_switch_plan"),
                getattr(self.server_args, "ddit_switch_plan", None),
            )

    @staticmethod
    def _ddit_is_warmup_req(req: Req | None) -> bool:
        return bool(getattr(req, "is_warmup", False))

    def _ddit_build_request_state(
        self,
        *,
        req: Req,
        request_id: str,
        num_timesteps: int,
        world_size: int,
        schedule_policy: str,
    ) -> DDiTRequestState:
        state = DDiTRequestState(
            request_id=request_id,
            resolution=resolve_resolution_key(req),
            total_steps=int(num_timesteps),
            arrival_time=(
                req.metrics.arrival_time_s
                if req.metrics and req.metrics.arrival_time_s
                else time.time()
            ),
            vae_k=int(getattr(self.server_args, "ddit_vae_gpus", 1)),
        )
        if schedule_policy == "forced_switch":
            plan = build_execution_plan(self.server_args, req, world_size=world_size)
            state.initial_ranks = plan.initial_ranks
            state.switch_plan = plan.switches
        return state

    def _ddit_has_compute_work(
        self,
        *,
        policy: Any,
        running_order: deque[str],
        pending_ops: list[deque[DDiTOp]],
    ) -> bool:
        if any(queue for queue in pending_ops):
            return True
        if hasattr(policy, "has_waiting_requests") and policy.has_waiting_requests():
            return True
        for request_id in running_order:
            req_state = policy.requests.get(request_id)
            if req_state and req_state.phase == RequestPhase.DIT:
                if req_state.cur_step < req_state.total_steps:
                    return True
        return False

    def _ddit_waiting_prepare_admitted(self, policy: Any, item: Any) -> bool:
        if not isinstance(item, Req) or item.is_warmup:
            return True
        candidate = DDiTRequestState(
            request_id=str(getattr(item, "request_id", "") or "pending"),
            resolution=resolve_resolution_key(item),
            total_steps=int(getattr(item, "num_inference_steps", 0) or 0),
        )
        if isinstance(policy, ForcedSwitchScheduler):
            plan = build_execution_plan(
                self.server_args,
                item,
                world_size=get_world_group().world_size,
            )
            candidate.initial_ranks = plan.initial_ranks
            candidate.switch_plan = plan.switches
        prepare_credit = getattr(policy, "prepare_credit", None)
        if prepare_credit is None:
            gpu_owner = getattr(policy, "gpu_owner", {})
            return any(owner is None for owner in gpu_owner.values())
        return int(prepare_credit(candidate)) > 0

    def _ddit_record_prepare_backpressure(
        self,
        *,
        wave_id: int,
        item: Any,
        reason: str,
    ) -> None:
        request_id = getattr(item, "request_id", None)
        now = time.time()
        record_op_trace_rows(
            self.server_args,
            [
                {
                    "timestamp_start": now,
                    "timestamp_end": now,
                    "request_id": request_id,
                    "stage": "prepare",
                    "step": None,
                    "ranks": [],
                    "wave_id": wave_id,
                    "op_id": f"wave{wave_id}:prepare_backpressure",
                    "rank": 0,
                    "action": "prepare_backpressure",
                    "status": "blocked",
                    "error": reason,
                }
            ],
        )

    def _ddit_policy_prepare_credit(self, policy: Any) -> int:
        prepare_credit = getattr(policy, "prepare_credit", None)
        if prepare_credit is None:
            gpu_owner = getattr(policy, "gpu_owner", {})
            return 1 if any(owner is None for owner in gpu_owner.values()) else 0
        return max(0, int(prepare_credit(None)))

    def _ddit_send_worker_credit(self, policy: Any) -> None:
        if self.gpu_id != 0 or self._pool_result_push is None:
            return
        credit = self._ddit_policy_prepare_credit(policy)
        msg = TransferCreditMsg(
            role=RoleType.DDIT_WORKER.value,
            instance_id=int(getattr(self.server_args, "disagg_instance_id", 0)),
            free_slots=credit,
            capacity_slots=max(1, int(getattr(self.server_args, "ddit_window_size", 1))),
        )
        self._pool_result_push.send_multipart(encode_transfer_msg(msg))

    def _ddit_drain_disagg_prepared(
        self,
        *,
        pending_register: deque[DDiTOp],
        registering: set[str],
        full_ranks: tuple[int, ...],
    ) -> bool:
        if self.gpu_id != 0 or self._compute_ready_queue is None:
            return False

        handled = False
        staging_backpressure = self._has_pending_outbound_staging_retry()
        handled |= self._process_transfer_control_queue(
            allow_new_work=not staging_backpressure
        )
        handled |= self._process_outbound_staging_retry_once()
        handled |= self._process_swap_out_queue_once()
        handled |= self._process_send_ready_queue_once()
        handled |= self._maybe_apply_pending_transfer_reconfigure()
        if staging_backpressure:
            return handled
        handled |= self._drain_disagg_work_socket() > 0
        handled |= self._process_transfer_control_queue()
        handled |= self._process_prefetch_queue_once()
        handled |= self._process_swapping_queue_once()

        while True:
            try:
                item = self._compute_ready_queue.get_nowait()
            except Exception:
                break
            if self._is_request_aborted(item.request_id):
                handled = True
                continue
            scalar_error = self._validate_inbound_scalar_fields(
                item.request_id, item.scalar_fields
            )
            if scalar_error is not None:
                self._fail_inbound_transfer(
                    item.request_id,
                    scalar_error,
                    item.prealloc_slot_id,
                )
                handled = True
                continue
            self._release_pending_receive(item.request_id, item.prealloc_slot_id)
            if item.request_id in registering:
                handled = True
                continue
            if item.scalar_fields.get("is_warmup"):
                inbound_sizes = self._warmup_inbound_sizes.pop(item.request_id, (0, 0))
                self._schedule_transfer_reconfigure(
                    inbound_sizes[0],
                    inbound_sizes[1],
                )
                reconfigured = self._maybe_apply_pending_transfer_reconfigure()
                logger.info(
                    "DDiT worker warmup calibration completed for %s "
                    "(inbound_data=%d bytes, inbound_meta=%d bytes, "
                    "reconfigured=%s); returning startup ACK",
                    item.request_id,
                    int(inbound_sizes[0]),
                    int(inbound_sizes[1]),
                    reconfigured,
                )
                logger.info(
                    "DDiT worker: queued warmup prepared request %s for "
                    "dynamic SP activation calibration (tensor_fields=%s, "
                    "scalar_fields=%d)",
                    item.request_id,
                    sorted(item.tensors.keys()),
                    len(item.scalar_fields),
                )
            pending_register.append(
                DDiTOp(
                    action="register_prepared",
                    request_id=item.request_id,
                    ranks=full_ranks,
                    stage="prepare",
                    payload={
                        "scalar_fields": item.scalar_fields,
                        "tensors": item.tensors,
                    },
                )
            )
            registering.add(item.request_id)
            logger.info(
                "DDiT worker: queued prepared request %s for policy registration "
                "(tensor_fields=%s, scalar_fields=%d)",
                item.request_id,
                sorted(item.tensors.keys()),
                len(item.scalar_fields),
            )
            handled = True
        return handled

    def _ddit_send_output_to_disagg_server(
        self,
        request_id: str,
        output_batch: OutputBatch,
    ) -> None:
        if self._pool_result_push is None:
            return
        tensor_fields = {}
        scalar_fields = {"request_id": request_id}
        if output_batch.output is not None:
            tensor_fields["output"] = output_batch.output
        if output_batch.audio is not None:
            tensor_fields["audio"] = output_batch.audio
        if output_batch.audio_sample_rate is not None:
            scalar_fields["audio_sample_rate"] = output_batch.audio_sample_rate
        if output_batch.error is not None:
            scalar_fields["error"] = output_batch.error
        if output_batch.output_file_paths:
            output_file_paths = [
                str(path) for path in output_batch.output_file_paths if path
            ]
            if output_file_paths:
                scalar_fields["output_file_paths"] = output_file_paths
        send_tensors(self._pool_result_push, tensor_fields, scalar_fields)

    def _ddit_queue_dynamic_sp_ensure(
        self,
        *,
        pending_dynamic_sp: deque[DDiTOp],
        ensured_dynamic_sp: set[tuple[int, ...]],
        ensuring_dynamic_sp: set[tuple[int, ...]],
        full_ranks: tuple[int, ...],
        target_ranks: tuple[int, ...],
        request_id: str | None,
        reason: str,
    ) -> None:
        target_ranks = tuple(sorted(int(rank) for rank in target_ranks))
        registry = get_dynamic_sp_registry(self.server_args)
        spec = registry.resolve_spec(target_ranks)
        if target_ranks in ensured_dynamic_sp or target_ranks in ensuring_dynamic_sp:
            return
        if registry.has(target_ranks):
            ensured_dynamic_sp.add(target_ranks)
            logger.debug(
                "DDiT worker: dynamic SP cache hit for ranks=%s "
                "(degree_pair=%sx%s, reason=%s); skipping ensure wave",
                spec.ranks,
                spec.ulysses_degree,
                spec.ring_degree,
                reason,
            )
            return
        pending_dynamic_sp.append(
            DDiTOp(
                action="ensure_dynamic_sp",
                request_id=request_id,
                ranks=full_ranks,
                stage="ddit_sp",
                payload={
                    "target_ranks": spec.ranks,
                    "log_reason": reason,
                },
                op_id=f"ensure_dynamic_sp:{','.join(str(r) for r in target_ranks)}",
            )
        )
        ensuring_dynamic_sp.add(target_ranks)
        logger.info(
            "DDiT worker: dynamic SP cache miss for ranks=%s "
            "(degree_pair=%sx%s, reason=%s); queued full-rank ensure",
            spec.ranks,
            spec.ulysses_degree,
            spec.ring_degree,
            reason,
        )

    def _ddit_activation_key(
        self, *, req: Req, target_ranks: tuple[int, ...]
    ) -> tuple[Any, ...]:
        registry = get_dynamic_sp_registry(self.server_args)
        spec = registry.resolve_spec(target_ranks)

        def shape_signature(value: Any) -> tuple[Any, ...]:
            shape = getattr(value, "shape", None)
            if shape is not None:
                return tuple(int(dim) for dim in shape)
            if isinstance(value, (list, tuple)):
                if all(
                    not hasattr(item, "shape") and not isinstance(item, (list, tuple))
                    for item in value
                ):
                    return tuple(value)
                return tuple(shape_signature(item) for item in value)
            return ()

        extra = getattr(req, "extra", None)
        if not isinstance(extra, dict):
            extra = {}
            try:
                req.extra = extra
            except Exception:
                pass
        static_signature = extra.get("ddit_activation_static_signature")
        if static_signature is None:
            latents = getattr(req, "latents", None)
            image_latent = getattr(req, "image_latent", None)
            num_steps = extra.get(
                "cache_dit_num_inference_steps",
                getattr(req, "num_inference_steps", None),
            )
            static_signature = (
                getattr(req, "height", None),
                getattr(req, "width", None),
                getattr(req, "num_frames", None),
                num_steps,
                shape_signature(getattr(req, "raw_latent_shape", None)),
                shape_signature(latents),
                shape_signature(image_latent),
                str(getattr(latents, "dtype", "unknown")),
                bool(getattr(req, "do_classifier_free_guidance", False)),
            )
            extra["ddit_activation_static_signature"] = static_signature
        return (
            "coarse",
            len(spec.ranks),
            spec.ulysses_degree,
            spec.ring_degree,
            resolve_resolution_key(req),
            tuple(static_signature),
            str(getattr(self.server_args, "ddit_profile_model_id", None) or ""),
            str(getattr(self.server_args, "model_id", None) or ""),
            str(getattr(self.server_args, "ddit_sp_degree_map", None) or ""),
        )

    def _ddit_queue_dynamic_sp_activation(
        self,
        *,
        pending_dynamic_sp: deque[DDiTOp],
        activated_dynamic_sp: set[tuple[Any, ...]],
        activating_dynamic_sp: set[tuple[Any, ...]],
        target_ranks: tuple[int, ...],
        request_id: str,
        req: Req,
        reason: str,
        begin_index: int = 0,
        force: bool = False,
    ) -> None:
        target_ranks = tuple(sorted(int(rank) for rank in target_ranks))
        activation_key = self._ddit_activation_key(req=req, target_ranks=target_ranks)
        if not force and (
            activation_key in activated_dynamic_sp
            or activation_key in activating_dynamic_sp
        ):
            return
        pending_dynamic_sp.append(
            DDiTOp(
                action="activate_dynamic_sp",
                request_id=request_id,
                ranks=target_ranks,
                stage="ddit_sp",
                step=begin_index,
                payload={
                    "activation_key": activation_key,
                    "begin_index": begin_index,
                    "log_reason": reason,
                    "force_activation": force,
                },
                op_id=f"activate_dynamic_sp:{','.join(str(r) for r in target_ranks)}",
            )
        )
        if not force:
            activating_dynamic_sp.add(activation_key)
        logger.info(
            "DDiT worker: queued dynamic SP activation for ranks=%s "
            "(reason=%s, begin_index=%s, force=%s)",
            target_ranks,
            reason,
            begin_index,
            force,
        )

    def _ddit_queue_request_plan_dynamic_sp_ensures(
        self,
        *,
        req: Req,
        world_size: int,
        state: DDiTRequestState,
        pending_dynamic_sp: deque[DDiTOp],
        ensured_dynamic_sp: set[tuple[int, ...]],
        ensuring_dynamic_sp: set[tuple[int, ...]],
        activated_dynamic_sp: set[tuple[Any, ...]],
        activating_dynamic_sp: set[tuple[Any, ...]],
        full_ranks: tuple[int, ...],
    ) -> None:
        plan_ranks: list[tuple[int, ...]] = []
        if state.initial_ranks:
            plan_ranks.append(tuple(state.initial_ranks))
        plan_ranks.extend(tuple(event.ranks) for event in state.switch_plan)
        activation_ranks: list[tuple[int, ...]] = list(plan_ranks)
        is_warmup_req = self._ddit_is_warmup_req(req)
        if is_warmup_req:
            prebuild_ranks = list(
                prebuild_rank_tuples_for_server(self.server_args, world_size)
            )
            coverage_ranks = list(
                activation_rank_tuples_for_server(self.server_args, world_size)
            )
            plan_ranks.extend(prebuild_ranks)
            activation_ranks = coverage_ranks
        if not plan_ranks and not activation_ranks:
            return

        ensure_ranks = sorted(
            set(plan_ranks).union(activation_ranks),
            key=lambda value: (len(value), value),
        )
        if is_warmup_req:
            logger.info(
                "DDiT worker: warmup dynamic SP plan for %s "
                "(rank_specs=%s, activation_warmups=%s, mode=%s)",
                state.request_id,
                len(ensure_ranks),
                len(set(activation_ranks)),
                getattr(self.server_args, "ddit_dynamic_sp_prebuild_mode", "auto"),
            )
        log_reason = "startup_warmup" if is_warmup_req else "request_plan"
        for ranks in ensure_ranks:
            self._ddit_queue_dynamic_sp_ensure(
                pending_dynamic_sp=pending_dynamic_sp,
                ensured_dynamic_sp=ensured_dynamic_sp,
                ensuring_dynamic_sp=ensuring_dynamic_sp,
                full_ranks=full_ranks,
                target_ranks=ranks,
                request_id=state.request_id,
                reason=log_reason,
            )
        for ranks in sorted(set(activation_ranks), key=lambda value: (len(value), value)):
            self._ddit_queue_dynamic_sp_activation(
                pending_dynamic_sp=pending_dynamic_sp,
                activated_dynamic_sp=activated_dynamic_sp,
                activating_dynamic_sp=activating_dynamic_sp,
                target_ranks=ranks,
                request_id=state.request_id,
                req=req,
                reason=log_reason,
                begin_index=0,
                force=is_warmup_req,
            )

    def _ddit_enqueue_schedule_decisions(
        self,
        *,
        policy: Any,
        schedule_policy: str,
        pending_dynamic_sp: deque[DDiTOp],
        pending_init: deque[DDiTOp],
        pending_migrate: deque[DDiTOp],
        initializing: set[str],
        migrating: set[str],
        req_by_id: dict[str, Req],
        ensured_dynamic_sp: set[tuple[int, ...]],
        ensuring_dynamic_sp: set[tuple[int, ...]],
        activated_dynamic_sp: set[tuple[Any, ...]],
        activating_dynamic_sp: set[tuple[Any, ...]],
        full_ranks: tuple[int, ...],
    ) -> None:
        for decision in policy.schedule():
            request_id = decision["request_id"]
            if decision["old_ranks"]:
                if request_id in migrating:
                    continue
                old_ranks = tuple(decision["old_ranks"])
                new_ranks = tuple(decision["new_ranks"])
                step = int(policy.requests[request_id].cur_step)
                req = req_by_id[request_id]
                if not self._ddit_is_warmup_req(req):
                    record_rank_switch(
                        self.server_args,
                        req,
                        stage="dit",
                        step=step,
                        old_ranks=old_ranks,
                        new_ranks=new_ranks,
                        reason=str(decision.get("reason", schedule_policy)),
                        policy=str(decision.get("policy", schedule_policy)),
                    )
                logger.info(
                    "DDiT worker: scheduled DiT migrate for %s at step=%d "
                    "(old_ranks=%s, new_ranks=%s, reason=%s, policy=%s)",
                    request_id,
                    step,
                    old_ranks,
                    new_ranks,
                    decision.get("reason", schedule_policy),
                    decision.get("policy", schedule_policy),
                )
                self._ddit_queue_dynamic_sp_ensure(
                    pending_dynamic_sp=pending_dynamic_sp,
                    ensured_dynamic_sp=ensured_dynamic_sp,
                    ensuring_dynamic_sp=ensuring_dynamic_sp,
                    full_ranks=full_ranks,
                    target_ranks=new_ranks,
                    request_id=request_id,
                    reason="dit_migrate",
                )
                self._ddit_queue_dynamic_sp_activation(
                    pending_dynamic_sp=pending_dynamic_sp,
                    activated_dynamic_sp=activated_dynamic_sp,
                    activating_dynamic_sp=activating_dynamic_sp,
                    target_ranks=new_ranks,
                    request_id=request_id,
                    req=req,
                    reason="dit_migrate",
                    begin_index=step,
                )
                pending_migrate.append(
                    DDiTOp(
                        action="dit_migrate",
                        request_id=request_id,
                        ranks=tuple(sorted(set(old_ranks) | set(new_ranks))),
                        stage="dit",
                        step=step,
                        payload={
                            "old_ranks": old_ranks,
                            "new_ranks": new_ranks,
                        },
                    )
                )
                migrating.add(request_id)
                continue

            if request_id in initializing:
                continue
            ranks = tuple(decision["new_ranks"])
            req = req_by_id[request_id]
            log_stage = "baseline" if decision.get("stage") == "baseline" else "dit"
            log_reason = str(decision.get("reason", schedule_policy))
            log_policy = str(decision.get("policy", schedule_policy))
            if not self._ddit_is_warmup_req(req):
                record_lifecycle(self.server_args, req, "dit_start")
                record_rank_switch(
                    self.server_args,
                    req,
                    stage=log_stage,
                    step=None,
                    old_ranks=(),
                    new_ranks=ranks,
                    reason=log_reason,
                    policy=log_policy,
                )
            self._ddit_queue_dynamic_sp_ensure(
                pending_dynamic_sp=pending_dynamic_sp,
                ensured_dynamic_sp=ensured_dynamic_sp,
                ensuring_dynamic_sp=ensuring_dynamic_sp,
                full_ranks=full_ranks,
                target_ranks=ranks,
                request_id=request_id,
                reason="dit_init",
            )
            self._ddit_queue_dynamic_sp_activation(
                pending_dynamic_sp=pending_dynamic_sp,
                activated_dynamic_sp=activated_dynamic_sp,
                activating_dynamic_sp=activating_dynamic_sp,
                target_ranks=ranks,
                request_id=request_id,
                req=req,
                reason="dit_init",
                begin_index=0,
            )
            pending_init.append(
                DDiTOp(
                    action="dit_init",
                    request_id=request_id,
                    ranks=ranks,
                    stage=log_stage,
                    payload={
                        "log_stage": log_stage,
                        "log_reason": log_reason,
                        "log_policy": log_policy,
                    },
                )
            )
            initializing.add(request_id)

    def _ddit_add_queued_ops(
        self, builder: CommandWaveBuilder, queue: deque[DDiTOp]
    ) -> None:
        retained: deque[DDiTOp] = deque()
        while queue:
            op = queue.popleft()
            if not builder.add(op):
                retained.append(op)
        queue.extend(retained)

    def _ddit_add_step_ops(
        self,
        *,
        builder: CommandWaveBuilder,
        policy: Any,
        running_order: deque[str],
        blocked: set[str],
    ) -> None:
        for _ in range(len(running_order)):
            request_id = running_order.popleft()
            req_state = policy.requests.get(request_id)
            if req_state is None or req_state.phase != RequestPhase.DIT:
                continue
            if request_id in blocked:
                running_order.append(request_id)
                continue
            if req_state.cur_step >= req_state.total_steps:
                running_order.append(request_id)
                continue
            op = DDiTOp(
                action="dit_step",
                request_id=request_id,
                ranks=req_state.ranks,
                stage="dit",
                step=req_state.cur_step,
            )
            if builder.add(op):
                running_order.append(request_id)
            else:
                running_order.append(request_id)

    def _ddit_build_compute_wave(
        self,
        *,
        wave_id: int,
        world_size: int,
        policy: Any,
        running_order: deque[str],
        pending_register: deque[DDiTOp],
        pending_dynamic_sp: deque[DDiTOp],
        pending_output_transfer: deque[DDiTOp],
        pending_vae_run: deque[DDiTOp],
        pending_vae_prepare: deque[DDiTOp],
        pending_finish: deque[DDiTOp],
        pending_migrate: deque[DDiTOp],
        pending_init: deque[DDiTOp],
        blocked: set[str],
    ) -> CommandWave:
        builder = CommandWaveBuilder(wave_id, world_size)
        for queue in (
            pending_dynamic_sp,
            pending_register,
            pending_output_transfer,
            pending_vae_run,
            pending_vae_prepare,
            pending_finish,
            pending_migrate,
            pending_init,
        ):
            self._ddit_add_queued_ops(builder, queue)
        self._ddit_add_step_ops(
            builder=builder,
            policy=policy,
            running_order=running_order,
            blocked=blocked,
        )
        return builder.build()

    def _hungry_broadcast_command(self, command: dict[str, Any] | None) -> dict[str, Any]:
        world = get_world_group()
        payload = [command] if self.gpu_id == 0 else None
        broadcasted = broadcast_pyobj(
            payload,
            world.rank,
            world.cpu_group,
            src=0,
        )
        return broadcasted[0]

    def _hungry_recv_rank0_reqs(self) -> list[tuple[bytes, Any]]:
        if self.receiver is None:
            return []
        recv_reqs: list[tuple[bytes, Any]] = []
        while True:
            try:
                parts = self.receiver.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            identity, payload = parts[0], parts[-1]
            try:
                reqs = pickle.loads(payload) if len(parts) > 2 else []
            except (pickle.UnpicklingError, EOFError):
                continue
            if not isinstance(reqs, list):
                reqs = [reqs]
            recv_reqs.extend((identity, req) for req in reqs)
        if recv_reqs:
            recv_reqs = self.process_received_reqs_with_req_based_warmup(recv_reqs)
            now_s = time.time()
            for _identity, req in recv_reqs:
                if (
                    isinstance(req, Req)
                    and not req.is_warmup
                    and req.metrics is not None
                    and req.metrics.arrival_time_s is None
                ):
                    req.metrics.arrival_time_s = now_s
                if isinstance(req, Req) and not req.is_warmup:
                    record_lifecycle(
                        self.server_args,
                        req,
                        "add",
                        timestamp=now_s,
                        status="queued",
                    )
            self.waiting_queue.extend(recv_reqs)
        return recv_reqs

    def _hungry_run_command(self, command: dict[str, Any]) -> Any:
        action = command["action"]
        if action == "idle":
            time.sleep(command.get("sleep_s", 0.01))
            return {"status": "idle"}
        if action == "prepare":
            return self.worker.prepare_hungry_request(command["req"])
        if action == "start_dit":
            return self.worker.start_hungry_dit(
                command["request_id"], tuple(command["ranks"])
            )
        if action == "dit_step":
            return self.worker.run_hungry_dit_step(
                command["request_id"], tuple(command["ranks"])
            )
        if action == "finish_dit":
            return self.worker.finish_hungry_dit(command["request_id"])
        if action == "vae":
            return self.worker.run_hungry_vae(
                command["request_id"], tuple(command["ranks"])
            )
        if action == "forward":
            return self.worker.execute_forward([command["req"]])
        if action == "control":
            req = command["req"]
            handler = self.request_handlers.get(type(req))
            if handler is None:
                return OutputBatch(error=f"Unknown request type: {type(req)}")
            return handler([req])
        if action == "shutdown":
            self._running = False
            return OutputBatch()
        raise ValueError(f"Unknown hungry command action: {action}")

    def _hungry_next_running_request(
        self,
        policy: HungryFirstScheduler,
        running_order: deque[str],
        finish_queue: deque[str],
    ) -> str | None:
        for _ in range(len(running_order)):
            request_id = running_order.popleft()
            req_state = policy.requests.get(request_id)
            if req_state is None or req_state.phase != RequestPhase.DIT:
                continue
            if req_state.cur_step >= req_state.total_steps:
                finish_queue.append(request_id)
                continue
            running_order.append(request_id)
            return request_id
        return None

    def _hungry_fail_request(
        self,
        *,
        request_id: str,
        req: Req | None,
        identity: bytes | None,
        error: str,
        policy: HungryFirstScheduler,
    ) -> None:
        if req is not None:
            record_lifecycle(self.server_args, req, "vae_end", error=error)
        if request_id in policy.requests:
            state = policy.requests[request_id]
            for rank in state.ranks:
                policy.gpu_owner[rank] = None
            state.ranks = ()
            state.phase = RequestPhase.DONE
        if self.gpu_id == 0 and identity is not None:
            self.return_result(OutputBatch(error=error), identity, is_warmup=False)

    def _ddit_concurrent_event_loop(
        self, schedule_policy: str, *, disagg_prepared: bool = False
    ) -> None:
        logger.info(
            "Starting single-node concurrent DDiT event loop with policy=%s disagg_prepared=%s.",
            schedule_policy,
            disagg_prepared,
        )
        world_size = get_world_group().world_size
        full_ranks = tuple(range(world_size))
        if self.gpu_id != 0:
            self._ddit_slave_concurrent_loop()
            self._cleanup_disagg()
            self.context.destroy(linger=0)
            return

        if schedule_policy == "fixed_baseline":
            policy = FixedBaselineScheduler(
                build_fixed_baseline_scheduler_config(self.server_args, world_size)
            )
        elif schedule_policy == "forced_switch":
            policy = ForcedSwitchScheduler(
                build_forced_switch_scheduler_config(self.server_args, world_size)
            )
        elif schedule_policy == "hungry_first":
            policy = HungryFirstScheduler(
                build_hungry_scheduler_config(self.server_args, world_size)
            )
        else:
            policy = build_profile_scheduler(
                schedule_policy, self.server_args, world_size
            )

        identities: dict[str, bytes | None] = {}
        req_by_id: dict[str, Req] = {}
        running_order: deque[str] = deque()
        pending_init: deque[DDiTOp] = deque()
        pending_register: deque[DDiTOp] = deque()
        pending_dynamic_sp: deque[DDiTOp] = deque()
        pending_migrate: deque[DDiTOp] = deque()
        pending_finish: deque[DDiTOp] = deque()
        pending_vae_prepare: deque[DDiTOp] = deque()
        pending_vae_run: deque[DDiTOp] = deque()
        pending_output_transfer: deque[DDiTOp] = deque()
        initializing: set[str] = set()
        migrating: set[str] = set()
        finishing: set[str] = set()
        vae_preparing: set[str] = set()
        vae_running: set[str] = set()
        output_transferring: set[str] = set()
        registering: set[str] = set()
        ensured_dynamic_sp: set[tuple[int, ...]] = set()
        ensuring_dynamic_sp: set[tuple[int, ...]] = set()
        activated_dynamic_sp: set[tuple[Any, ...]] = set()
        activating_dynamic_sp: set[tuple[Any, ...]] = set()
        warmup_activation_summary: dict[str, dict[str, Any]] = {}
        prepare_backpressure_logged: set[str] = set()
        prepared_since_compute = False
        wave_id = 0

        while self._running:
            if disagg_prepared:
                self._ddit_send_worker_credit(policy)
                self._ddit_drain_disagg_prepared(
                    pending_register=pending_register,
                    registering=registering,
                    full_ranks=full_ranks,
                )
            else:
                self._hungry_recv_rank0_reqs()
            pending_queues = [
                pending_register,
                pending_dynamic_sp,
                pending_init,
                pending_migrate,
                pending_finish,
                pending_vae_prepare,
                pending_vae_run,
                pending_output_transfer,
            ]
            compute_ready = self._ddit_has_compute_work(
                policy=policy,
                running_order=running_order,
                pending_ops=pending_queues,
            )

            wave: CommandWave
            can_consider_prepare = (
                not disagg_prepared
                and self.waiting_queue
                and (not compute_ready or not prepared_since_compute)
            )
            if can_consider_prepare and self._ddit_waiting_prepare_admitted(
                policy, self.waiting_queue[0][1]
            ):
                identity, item = self.waiting_queue.popleft()
                prepare_backpressure_logged.discard(
                    str(getattr(item, "request_id", None) or id(item))
                )
                if isinstance(item, Req):
                    action = "full_forward" if item.is_warmup else "full_prepare"
                    wave = self._ddit_exclusive_wave(
                        wave_id=wave_id,
                        action=action,
                        ranks=full_ranks,
                        payload={"req": item},
                        request_id=getattr(item, "request_id", None),
                        stage="prepare" if not item.is_warmup else "warmup",
                    )
                    command_identity = identity
                elif isinstance(item, ShutdownReq):
                    wave = self._ddit_exclusive_wave(
                        wave_id=wave_id,
                        action="shutdown",
                        ranks=full_ranks,
                        payload={},
                        stage="shutdown",
                    )
                    command_identity = identity
                else:
                    wave = self._ddit_exclusive_wave(
                        wave_id=wave_id,
                        action="control",
                        ranks=full_ranks,
                        payload={"req": item},
                        stage="control",
                    )
                    command_identity = identity
            elif can_consider_prepare:
                blocked_item = self.waiting_queue[0][1]
                blocked_request_id = str(
                    getattr(blocked_item, "request_id", None) or id(blocked_item)
                )
                if blocked_request_id not in prepare_backpressure_logged:
                    self._ddit_record_prepare_backpressure(
                        wave_id=wave_id,
                        item=blocked_item,
                        reason="no_dit_vae_prepare_credit",
                    )
                    prepare_backpressure_logged.add(blocked_request_id)
                self._ddit_enqueue_schedule_decisions(
                    policy=policy,
                    schedule_policy=schedule_policy,
                    pending_dynamic_sp=pending_dynamic_sp,
                    pending_init=pending_init,
                    pending_migrate=pending_migrate,
                    initializing=initializing,
                    migrating=migrating,
                    req_by_id=req_by_id,
                    ensured_dynamic_sp=ensured_dynamic_sp,
                    ensuring_dynamic_sp=ensuring_dynamic_sp,
                    activated_dynamic_sp=activated_dynamic_sp,
                    activating_dynamic_sp=activating_dynamic_sp,
                    full_ranks=full_ranks,
                )
                blocked = (
                    registering
                    | initializing
                    | migrating
                    | finishing
                    | vae_preparing
                    | vae_running
                    | output_transferring
                )
                wave = self._ddit_build_compute_wave(
                    wave_id=wave_id,
                    world_size=world_size,
                    policy=policy,
                    running_order=running_order,
                    pending_register=pending_register,
                    pending_dynamic_sp=pending_dynamic_sp,
                    pending_output_transfer=pending_output_transfer,
                    pending_vae_run=pending_vae_run,
                    pending_vae_prepare=pending_vae_prepare,
                    pending_finish=pending_finish,
                    pending_migrate=pending_migrate,
                    pending_init=pending_init,
                    blocked=blocked,
                )
                command_identity = None
                if not wave.ops:
                    wave = CommandWave(wave_id, ())
                    time.sleep(0.01)
            else:
                self._ddit_enqueue_schedule_decisions(
                    policy=policy,
                    schedule_policy=schedule_policy,
                    pending_dynamic_sp=pending_dynamic_sp,
                    pending_init=pending_init,
                    pending_migrate=pending_migrate,
                    initializing=initializing,
                    migrating=migrating,
                    req_by_id=req_by_id,
                    ensured_dynamic_sp=ensured_dynamic_sp,
                    ensuring_dynamic_sp=ensuring_dynamic_sp,
                    activated_dynamic_sp=activated_dynamic_sp,
                    activating_dynamic_sp=activating_dynamic_sp,
                    full_ranks=full_ranks,
                )
                blocked = (
                    registering
                    | initializing
                    | migrating
                    | finishing
                    | vae_preparing
                    | vae_running
                    | output_transferring
                )
                wave = self._ddit_build_compute_wave(
                    wave_id=wave_id,
                    world_size=world_size,
                    policy=policy,
                    running_order=running_order,
                    pending_register=pending_register,
                    pending_dynamic_sp=pending_dynamic_sp,
                    pending_output_transfer=pending_output_transfer,
                    pending_vae_run=pending_vae_run,
                    pending_vae_prepare=pending_vae_prepare,
                    pending_finish=pending_finish,
                    pending_migrate=pending_migrate,
                    pending_init=pending_init,
                    blocked=blocked,
                )
                command_identity = None
                if not wave.ops:
                    wave = CommandWave(wave_id, ())
                    time.sleep(0.01)

            try:
                results = self._ddit_run_wave(wave, world_size)
                wave_id += 1
                has_compute_op = any(
                    op.action
                    not in (
                        "idle",
                        "full_prepare",
                        "register_prepared",
                        "ensure_dynamic_sp",
                        "activate_dynamic_sp",
                        "warmup_ack",
                        "full_forward",
                        "control",
                    )
                    for op in wave.ops
                )
                if has_compute_op:
                    prepared_since_compute = False

                for op in wave.ops:
                    result = self._ddit_result_for_op(results, op)
                    action = op.action
                    request_id = op.request_id

                    if action == "full_prepare":
                        req = op.payload["req"]
                        request_id = result["request_id"]
                        identities[request_id] = command_identity
                        req_by_id[request_id] = req
                        state = self._ddit_build_request_state(
                            request_id=request_id,
                            req=req,
                            num_timesteps=int(result["num_timesteps"]),
                            world_size=world_size,
                            schedule_policy=schedule_policy,
                        )
                        policy.add_request(state)
                        if isinstance(policy, FixedBaselineScheduler):
                            policy.mark_text_encoder_done(request_id)
                        self._ddit_log_registered_request_state(
                            source="local",
                            req=req,
                            state=state,
                            schedule_policy=schedule_policy,
                        )
                        self._ddit_queue_request_plan_dynamic_sp_ensures(
                            req=req,
                            world_size=world_size,
                            state=state,
                            pending_dynamic_sp=pending_dynamic_sp,
                            ensured_dynamic_sp=ensured_dynamic_sp,
                            ensuring_dynamic_sp=ensuring_dynamic_sp,
                            activated_dynamic_sp=activated_dynamic_sp,
                            activating_dynamic_sp=activating_dynamic_sp,
                            full_ranks=full_ranks,
                        )
                        prepared_since_compute = True
                    elif action == "register_prepared":
                        registering.discard(request_id)
                        req = result["req"]
                        request_id = result["request_id"]
                        req_by_id[request_id] = req
                        state = self._ddit_build_request_state(
                            request_id=request_id,
                            req=req,
                            num_timesteps=int(result["num_timesteps"]),
                            world_size=world_size,
                            schedule_policy=schedule_policy,
                        )
                        if self._ddit_is_warmup_req(req):
                            self._ddit_log_registered_request_state(
                                source="warmup",
                                req=req,
                                state=state,
                                schedule_policy=schedule_policy,
                            )
                            self._ddit_queue_request_plan_dynamic_sp_ensures(
                                req=req,
                                world_size=world_size,
                                state=state,
                                pending_dynamic_sp=pending_dynamic_sp,
                                ensured_dynamic_sp=ensured_dynamic_sp,
                                ensuring_dynamic_sp=ensuring_dynamic_sp,
                                activated_dynamic_sp=activated_dynamic_sp,
                                activating_dynamic_sp=activating_dynamic_sp,
                                full_ranks=full_ranks,
                            )
                            pending_dynamic_sp.append(
                                DDiTOp(
                                    action="warmup_ack",
                                    request_id=request_id,
                                    ranks=full_ranks,
                                    stage="warmup",
                                    op_id=f"warmup_ack:{request_id}",
                                )
                            )
                            req_by_id.pop(request_id, None)
                            continue
                        policy.add_request(state)
                        if isinstance(policy, FixedBaselineScheduler):
                            policy.mark_text_encoder_done(request_id)
                        self._ddit_log_registered_request_state(
                            source="prepared",
                            req=req,
                            state=state,
                            schedule_policy=schedule_policy,
                        )
                        self._ddit_queue_request_plan_dynamic_sp_ensures(
                            req=req,
                            world_size=world_size,
                            state=state,
                            pending_dynamic_sp=pending_dynamic_sp,
                            ensured_dynamic_sp=ensured_dynamic_sp,
                            ensuring_dynamic_sp=ensuring_dynamic_sp,
                            activated_dynamic_sp=activated_dynamic_sp,
                            activating_dynamic_sp=activating_dynamic_sp,
                            full_ranks=full_ranks,
                        )
                    elif action == "ensure_dynamic_sp":
                        target_ranks = tuple(
                            int(rank)
                            for rank in (
                                result.get("target_ranks")
                                if isinstance(result, dict)
                                else op.payload.get("target_ranks", ())
                            )
                        )
                        if target_ranks:
                            ensuring_dynamic_sp.discard(target_ranks)
                            ensured_dynamic_sp.add(target_ranks)
                    elif action == "activate_dynamic_sp":
                        activation_key = tuple(
                            result.get("activation_key")
                            if isinstance(result, dict)
                            else op.payload.get("activation_key", ())
                        )
                        if activation_key:
                            activating_dynamic_sp.discard(activation_key)
                            activated_dynamic_sp.add(activation_key)
                        if isinstance(result, dict) and str(request_id).startswith(
                            "warmup-"
                        ):
                            summary = warmup_activation_summary.setdefault(
                                str(request_id),
                                {
                                    "count": 0,
                                    "force_count": 0,
                                    "cache_hits": 0,
                                    "activation_ms_total": 0.0,
                                    "model_forward_warmup_ms_total": 0.0,
                                    "activation_keys": set(),
                                },
                            )
                            summary["count"] += 1
                            summary["force_count"] += int(
                                bool(result.get("force_activation"))
                            )
                            summary["cache_hits"] += int(
                                bool(result.get("activation_cache_hit"))
                            )
                            summary["activation_ms_total"] += float(
                                result.get("activation_ms") or 0.0
                            )
                            summary["model_forward_warmup_ms_total"] += float(
                                result.get("model_forward_warmup_ms") or 0.0
                            )
                            if activation_key:
                                summary["activation_keys"].add(activation_key)
                    elif action == "warmup_ack":
                        summary = warmup_activation_summary.pop(
                            str(request_id), None
                        )
                        if summary is not None and self.gpu_id == 0:
                            logger.info(
                                "DDiT worker: startup warmup dynamic SP activation "
                                "done for %s (activation_ops=%s, force_ops=%s, "
                                "cache_hits=%s, unique_coarse_keys=%s, "
                                "activation_ms_total=%.2f, "
                                "model_forward_warmup_ms_total=%.2f)",
                                request_id,
                                summary["count"],
                                summary["force_count"],
                                summary["cache_hits"],
                                len(summary["activation_keys"]),
                                summary["activation_ms_total"],
                                summary["model_forward_warmup_ms_total"],
                            )
                    elif action == "full_forward":
                        req = op.payload["req"]
                        self._write_monolithic_profile_row(req, result)
                        self.return_result(
                            result, command_identity, is_warmup=req.is_warmup
                        )
                    elif action == "control":
                        self.return_result(result, command_identity, is_warmup=False)
                    elif action == "shutdown":
                        self._running = False
                    elif action == "dit_init":
                        initializing.discard(request_id)
                        if request_id not in running_order:
                            running_order.append(request_id)
                    elif action == "dit_migrate":
                        migrating.discard(request_id)
                    elif action == "dit_step":
                        cur_step = int(result["cur_step"])
                        self._ddit_policy_update_step(policy, request_id, cur_step)
                        if result["done"] and request_id not in finishing:
                            finishing.add(request_id)
                            req_state = policy.requests[request_id]
                            pending_finish.append(
                                DDiTOp(
                                    action="dit_finish",
                                    request_id=request_id,
                                    ranks=req_state.ranks,
                                    stage="dit",
                                    step=cur_step,
                                )
                            )
                    elif action == "dit_finish":
                        finishing.discard(request_id)
                        req = req_by_id[request_id]
                        req_state = policy.requests[request_id]
                        final_ranks = req_state.ranks
                        if not self._ddit_is_warmup_req(req):
                            record_lifecycle(self.server_args, req, "dit_end")
                        if getattr(policy, "vae_same_as_dit", False):
                            vae_ranks = final_ranks
                            req_state.phase = RequestPhase.VAE
                        else:
                            vae_ranks = resolve_vae_ranks(
                                self.server_args,
                                req,
                                world_size=world_size,
                                final_dit_ranks=final_ranks,
                            )
                            policy.transition_to_vae(request_id, vae_ranks)
                        if not self._ddit_is_warmup_req(req):
                            record_rank_switch(
                                self.server_args,
                                req,
                                stage="vae",
                                step=req_state.cur_step,
                                old_ranks=final_ranks,
                                new_ranks=vae_ranks,
                                reason=(
                                    f"{schedule_policy}_same_ranks"
                                    if getattr(policy, "vae_same_as_dit", False)
                                    else "dit_to_vae"
                                ),
                                policy=(
                                    schedule_policy
                                    if getattr(policy, "vae_same_as_dit", False)
                                    else "ddit_vae_gpus"
                                ),
                            )
                        vae_preparing.add(request_id)
                        pending_vae_prepare.append(
                            DDiTOp(
                                action="vae_prepare",
                                request_id=request_id,
                                ranks=tuple(sorted(set(final_ranks) | set(vae_ranks))),
                                stage="vae",
                                step=req_state.cur_step,
                                payload={
                                    "final_dit_ranks": final_ranks,
                                    "vae_ranks": vae_ranks,
                                },
                            )
                        )
                    elif action == "vae_prepare":
                        vae_preparing.discard(request_id)
                        req_state = policy.requests[request_id]
                        vae_running.add(request_id)
                        req = req_by_id[request_id]
                        if not self._ddit_is_warmup_req(req):
                            record_lifecycle(self.server_args, req, "vae_start")
                        pending_vae_run.append(
                            DDiTOp(
                                action="vae_run",
                                request_id=request_id,
                                ranks=req_state.ranks,
                                stage="vae",
                                step=req_state.cur_step,
                            )
                        )
                    elif action == "vae_run":
                        vae_running.discard(request_id)
                        leader = op.ranks[0]
                        if leader == 0:
                            self._ddit_policy_complete(policy, request_id)
                            req = req_by_id[request_id]
                            if not self._ddit_is_warmup_req(req):
                                record_lifecycle(self.server_args, req, "vae_end")
                            self._write_monolithic_profile_row(
                                req, result
                            )
                            if disagg_prepared:
                                self._ddit_send_output_to_disagg_server(
                                    request_id, result
                                )
                            else:
                                self.return_result(
                                    result,
                                    identities.get(request_id),
                                    is_warmup=False,
                                )
                            identities.pop(request_id, None)
                            req_by_id.pop(request_id, None)
                        else:
                            output_transferring.add(request_id)
                            pending_output_transfer.append(
                                DDiTOp(
                                    action="output_transfer",
                                    request_id=request_id,
                                    ranks=(0, leader),
                                    stage="output",
                                    step=op.step,
                                    payload={"src_rank": leader},
                                )
                            )
                    elif action == "output_transfer":
                        output_transferring.discard(request_id)
                        self._ddit_policy_complete(policy, request_id)
                        req = req_by_id[request_id]
                        if not self._ddit_is_warmup_req(req):
                            record_lifecycle(self.server_args, req, "vae_end")
                        self._write_monolithic_profile_row(
                            req, result
                        )
                        if disagg_prepared:
                            self._ddit_send_output_to_disagg_server(request_id, result)
                        else:
                            self.return_result(
                                result,
                                identities.get(request_id),
                                is_warmup=False,
                            )
                        identities.pop(request_id, None)
                        req_by_id.pop(request_id, None)
            except Exception as e:
                logger.error("Concurrent DDiT event loop failed: %s", e, exc_info=True)
                try:
                    self._ddit_fail_concurrent_wave_requests(
                        wave=wave,
                        error=str(e),
                        policy=policy,
                        running_order=running_order,
                        pending_queues=[
                            pending_register,
                            pending_dynamic_sp,
                            pending_init,
                            pending_migrate,
                            pending_finish,
                            pending_vae_prepare,
                            pending_vae_run,
                            pending_output_transfer,
                        ],
                        tracking_sets=[
                            registering,
                            initializing,
                            migrating,
                            finishing,
                            vae_preparing,
                            vae_running,
                            output_transferring,
                        ],
                        ensuring_dynamic_sp=ensuring_dynamic_sp,
                        activating_dynamic_sp=activating_dynamic_sp,
                        identities=identities,
                        req_by_id=req_by_id,
                        disagg_prepared=disagg_prepared,
                    )
                except Exception:
                    logger.debug(
                        "Ignoring concurrent DDiT request cleanup failure",
                        exc_info=True,
                    )
                # Keep the process alive for transient request-level failures.
                # Collective failures may still require process restart.
                time.sleep(0.05)

        shutdown_wave = self._ddit_exclusive_wave(
            wave_id=wave_id,
            action="shutdown",
            ranks=full_ranks,
            payload={},
            stage="shutdown",
        )
        try:
            self._ddit_run_wave(shutdown_wave, world_size)
        except Exception:
            logger.debug("Ignoring concurrent DDiT shutdown wave failure", exc_info=True)
        if self.receiver is not None:
            self.receiver.close()
        self._cleanup_disagg()
        self.context.destroy(linger=0)

    def _hungry_event_loop(self) -> None:
        logger.info("Starting single-node DDiT hungry-first event loop.")
        world_size = get_world_group().world_size
        policy = HungryFirstScheduler(
            build_hungry_scheduler_config(self.server_args, world_size)
        )
        identities: dict[str, bytes | None] = {}
        req_by_id: dict[str, Req] = {}
        running_order: deque[str] = deque()
        finish_queue: deque[str] = deque()
        vae_queue: deque[tuple[str, tuple[int, ...]]] = deque()
        started_dit: set[str] = set()

        while self._running:
            command: dict[str, Any] | None = None
            command_identity: bytes | None = None

            if self.gpu_id == 0:
                self._hungry_recv_rank0_reqs()

                if self.waiting_queue:
                    identity, item = self.waiting_queue.popleft()
                    if isinstance(item, Req):
                        command_identity = identity
                        command = {
                            "action": "forward" if item.is_warmup else "prepare",
                            "req": item,
                        }
                    elif isinstance(item, ShutdownReq):
                        command = {"action": "shutdown"}
                    else:
                        command_identity = identity
                        command = {"action": "control", "req": item}

                if command is None and finish_queue:
                    request_id = finish_queue.popleft()
                    command = {"action": "finish_dit", "request_id": request_id}

                if command is None and vae_queue:
                    request_id, vae_ranks = vae_queue.popleft()
                    command = {
                        "action": "vae",
                        "request_id": request_id,
                        "ranks": vae_ranks,
                    }

                if command is None:
                    decisions = policy.schedule()
                    for decision in decisions:
                        request_id = decision["request_id"]
                        if decision["old_ranks"]:
                            continue
                        if request_id not in started_dit:
                            started_dit.add(request_id)
                            running_order.append(request_id)
                            command = {
                                "action": "start_dit",
                                "request_id": request_id,
                                "ranks": tuple(decision["new_ranks"]),
                            }
                            break

                if command is None:
                    request_id = self._hungry_next_running_request(
                        policy, running_order, finish_queue
                    )
                    if request_id is not None:
                        command = {
                            "action": "dit_step",
                            "request_id": request_id,
                            "ranks": policy.requests[request_id].ranks,
                        }

                if command is None:
                    command = {"action": "idle", "sleep_s": 0.01}

            command = self._hungry_broadcast_command(command)
            try:
                result = self._hungry_run_command(command)
                action = command["action"]
                if self.gpu_id != 0:
                    continue

                if action == "prepare":
                    req = command["req"]
                    request_id = result["request_id"]
                    identities[request_id] = command_identity
                    req_by_id[request_id] = req
                    policy.add_request(
                        DDiTRequestState(
                            request_id=request_id,
                            resolution=resolve_resolution_key(req),
                            total_steps=int(result["num_timesteps"]),
                            arrival_time=(
                                req.metrics.arrival_time_s
                                if req.metrics and req.metrics.arrival_time_s
                                else time.time()
                            ),
                            vae_k=int(getattr(self.server_args, "ddit_vae_gpus", 1)),
                        )
                    )
                elif action == "forward":
                    req = command["req"]
                    self._write_monolithic_profile_row(req, result)
                    self.return_result(result, command_identity, is_warmup=req.is_warmup)
                elif action == "control":
                    self.return_result(result, command_identity, is_warmup=False)
                elif action == "dit_step":
                    request_id = command["request_id"]
                    policy.update_cur_step(request_id, int(result["cur_step"]))
                    if result["done"]:
                        finish_queue.append(request_id)
                elif action == "finish_dit":
                    request_id = command["request_id"]
                    req = req_by_id[request_id]
                    final_ranks = policy.requests[request_id].ranks
                    vae_ranks = resolve_vae_ranks(
                        self.server_args,
                        req,
                        world_size=world_size,
                        final_dit_ranks=final_ranks,
                    )
                    policy.transition_to_vae(request_id, vae_ranks)
                    vae_queue.append((request_id, vae_ranks))
                elif action == "vae":
                    request_id = command["request_id"]
                    policy.complete_vae(request_id)
                    self._write_monolithic_profile_row(req_by_id[request_id], result)
                    self.return_result(
                        result,
                        identities.get(request_id),
                        is_warmup=False,
                    )
                    identities.pop(request_id, None)
                    req_by_id.pop(request_id, None)
                    started_dit.discard(request_id)
            except Exception as e:
                logger.error("Hungry DDiT command failed: %s", e, exc_info=True)
                if self.gpu_id == 0:
                    request_id = command.get("request_id")
                    req = req_by_id.get(request_id) if request_id else command.get("req")
                    identity = identities.get(request_id) if request_id else command_identity
                    self._hungry_fail_request(
                        request_id=request_id or getattr(req, "request_id", "unknown"),
                        req=req if isinstance(req, Req) else None,
                        identity=identity,
                        error=str(e),
                        policy=policy,
                    )

        if self.receiver is not None:
            self.receiver.close()
        self._cleanup_disagg()
        self.context.destroy(linger=0)
