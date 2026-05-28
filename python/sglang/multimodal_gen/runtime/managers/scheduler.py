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
from sglang.multimodal_gen.runtime.ddit.config import (
    resolve_resolution_key,
    resolve_schedule_policy,
    resolve_vae_ranks,
)
from sglang.multimodal_gen.runtime.ddit.concurrent import (
    CommandWave,
    CommandWaveBuilder,
    DDiTOp,
)
from sglang.multimodal_gen.runtime.ddit.logging import record_lifecycle
from sglang.multimodal_gen.runtime.ddit.logging import record_op_trace_rows
from sglang.multimodal_gen.runtime.ddit.logging import record_rank_switch
from sglang.multimodal_gen.runtime.ddit.scheduler import (
    DDiTRequestState,
    FixedBaselineScheduler,
    HungryFirstScheduler,
    RequestPhase,
    build_fixed_baseline_scheduler_config,
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
        # Pool mode: all roles use the pool event loop
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
            self.task_pipes_to_slaves[rank - 1].send(commands[rank])
        results = [self._ddit_execute_rank_command(commands[0])]
        for rank in range(1, world_size):
            results.append(self.result_pipes_from_slaves[rank - 1].recv())
        record_op_trace_rows(self.server_args, [result["trace"] for result in results])
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

    def _ddit_enqueue_schedule_decisions(
        self,
        *,
        policy: Any,
        schedule_policy: str,
        pending_init: deque[DDiTOp],
        pending_migrate: deque[DDiTOp],
        initializing: set[str],
        migrating: set[str],
        req_by_id: dict[str, Req],
    ) -> None:
        for decision in policy.schedule():
            request_id = decision["request_id"]
            if decision["old_ranks"]:
                if request_id in migrating:
                    continue
                old_ranks = tuple(decision["old_ranks"])
                new_ranks = tuple(decision["new_ranks"])
                step = int(policy.requests[request_id].cur_step)
                record_rank_switch(
                    self.server_args,
                    req_by_id[request_id],
                    stage="dit",
                    step=step,
                    old_ranks=old_ranks,
                    new_ranks=new_ranks,
                    reason=str(decision.get("reason", schedule_policy)),
                    policy=str(decision.get("policy", schedule_policy)),
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

    def _ddit_concurrent_event_loop(self, schedule_policy: str) -> None:
        logger.info(
            "Starting single-node concurrent DDiT event loop with policy=%s.",
            schedule_policy,
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
        prepared_since_compute = False
        wave_id = 0

        while self._running:
            self._hungry_recv_rank0_reqs()
            pending_queues = [
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
            if self.waiting_queue and (not compute_ready or not prepared_since_compute):
                identity, item = self.waiting_queue.popleft()
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
            else:
                self._ddit_enqueue_schedule_decisions(
                    policy=policy,
                    schedule_policy=schedule_policy,
                    pending_init=pending_init,
                    pending_migrate=pending_migrate,
                    initializing=initializing,
                    migrating=migrating,
                    req_by_id=req_by_id,
                )
                blocked = (
                    initializing
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
                    not in ("idle", "full_prepare", "full_forward", "control")
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
                        state = DDiTRequestState(
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
                        policy.add_request(state)
                        if isinstance(policy, FixedBaselineScheduler):
                            policy.mark_text_encoder_done(request_id)
                        prepared_since_compute = True
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
                        record_lifecycle(
                            self.server_args, req_by_id[request_id], "vae_start"
                        )
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
                            record_lifecycle(
                                self.server_args, req_by_id[request_id], "vae_end"
                            )
                            self._write_monolithic_profile_row(
                                req_by_id[request_id], result
                            )
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
                        record_lifecycle(
                            self.server_args, req_by_id[request_id], "vae_end"
                        )
                        self._write_monolithic_profile_row(
                            req_by_id[request_id], result
                        )
                        self.return_result(
                            result,
                            identities.get(request_id),
                            is_warmup=False,
                        )
                        identities.pop(request_id, None)
                        req_by_id.pop(request_id, None)
            except Exception as e:
                logger.error("Concurrent DDiT event loop failed: %s", e, exc_info=True)
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
