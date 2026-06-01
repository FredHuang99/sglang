"""A lightweight ShiftServe router used by validation and local dry runs.

The class in this module owns the request-life bookkeeping and scheduling
decisions. Real GPU deployments can replace the mock stage clients with HTTP
or ZMQ adapters while retaining the same scheduler, metrics, and flip logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from sglang.shiftserve.metrics import MetricsRecorder
from sglang.shiftserve.scheduler import (
    HysteresisFlipMonitor,
    InstanceRuntimeState,
    RequestEstimate,
    SchedulerMode,
    ShiftServeScheduler,
    StageKind,
)


@dataclass(frozen=True)
class ShiftServeRequest:
    request_id: str
    input_tokens: int
    output_tokens: int
    bin_name: str = "short"


@dataclass(frozen=True)
class ShiftServeResponse:
    request_id: str
    pe_instance: str
    diffusion_instance: str
    generated_tokens: int
    flip_direction: str | None = None


class MockStageClient:
    """Deterministic stage adapter for CPU validation and unit tests."""

    def run_pe(self, request: ShiftServeRequest) -> int:
        return request.output_tokens

    def run_stage(self, stage: StageKind, request: ShiftServeRequest) -> None:
        return None


class ShiftServeRouter:
    def __init__(
        self,
        *,
        instances: list[InstanceRuntimeState],
        scheduler: ShiftServeScheduler,
        metrics: MetricsRecorder,
        flip_monitor: HysteresisFlipMonitor,
        stage_client: MockStageClient | None = None,
    ):
        self.instances = {instance.instance_id: instance for instance in instances}
        self.scheduler = scheduler
        self.metrics = metrics
        self.flip_monitor = flip_monitor
        self.stage_client = stage_client or MockStageClient()

    def submit(self, request: ShiftServeRequest) -> ShiftServeResponse:
        self.metrics.mark(request.request_id, "system_enter")
        pe_selection = self.scheduler.select(
            self.instances.values(),
            RequestEstimate(
                request_id=request.request_id,
                stage=StageKind.PE,
                bin_tokens=request.output_tokens,
                input_tokens=request.input_tokens,
            ),
        )
        pe = self.instances[pe_selection.instance_id]
        self.scheduler.enqueue(pe, request.request_id)
        self.metrics.mark(
            request.request_id,
            "pe_enter",
            stage=StageKind.PE.value,
            instance_id=pe.instance_id,
            reason=pe_selection.fallback_reason,
        )
        generated = self.stage_client.run_pe(request)
        self.scheduler.token_estimator.record(pe.instance_id, generated)
        flip_direction = self.flip_monitor.record_completion(generated)
        self.metrics.mark(
            request.request_id,
            "pe_end",
            stage=StageKind.PE.value,
            instance_id=pe.instance_id,
            generated_tokens=generated,
        )

        diffusion_stage = (
            StageKind.DIT_VAE
            if any(i.kind == StageKind.DIT_VAE for i in self.instances.values())
            else StageKind.DIT
        )
        diffusion_selection = self.scheduler.select(
            self.instances.values(),
            RequestEstimate(
                request_id=request.request_id,
                stage=diffusion_stage,
                bin_tokens=request.output_tokens,
            ),
        )
        diffusion = self.instances[diffusion_selection.instance_id]
        self.scheduler.enqueue(diffusion, request.request_id)

        te_instance_id = self._paired_te_instance_id(diffusion)
        self.metrics.mark(
            request.request_id,
            "te_enter",
            stage=StageKind.TE.value,
            instance_id=te_instance_id,
            bundle_id=diffusion.instance_id if diffusion.kind == StageKind.DIT_VAE else None,
        )
        self.stage_client.run_stage(StageKind.TE, request)
        self.metrics.mark(
            request.request_id,
            "te_end",
            stage=StageKind.TE.value,
            instance_id=te_instance_id,
            bundle_id=diffusion.instance_id if diffusion.kind == StageKind.DIT_VAE else None,
        )
        if diffusion.kind == StageKind.DIT_VAE and diffusion.dit_vae_bundle is not None:
            bundle = diffusion.dit_vae_bundle
            dit_started = bundle.start_next_dit()
            dit_slot_id = dit_started[0] if dit_started is not None else None
            self.metrics.mark(
                request.request_id,
                "dit_enter",
                stage=StageKind.DIT.value,
                instance_id=diffusion.instance_id,
                bundle_id=bundle.bundle_id,
                dit_slot_id=dit_slot_id,
            )
            self.stage_client.run_stage(StageKind.DIT, request)
            if dit_started is not None:
                bundle.finish_dit(request.request_id)
            self.metrics.mark(
                request.request_id,
                "dit_end",
                stage=StageKind.DIT.value,
                instance_id=diffusion.instance_id,
                bundle_id=bundle.bundle_id,
                dit_slot_id=dit_slot_id,
            )
            vae_started = bundle.start_next_vae()
            vae_slot_id = vae_started[0] if vae_started is not None else None
            self.metrics.mark(
                request.request_id,
                "vae_enter",
                stage=StageKind.VAE.value,
                instance_id=diffusion.instance_id,
                bundle_id=bundle.bundle_id,
                vae_slot_id=vae_slot_id,
            )
            self.stage_client.run_stage(StageKind.VAE, request)
            if vae_started is not None:
                bundle.finish_vae(request.request_id)
            self.metrics.mark(
                request.request_id,
                "vae_end",
                stage=StageKind.VAE.value,
                instance_id=diffusion.instance_id,
                bundle_id=bundle.bundle_id,
                vae_slot_id=vae_slot_id,
            )
        else:
            self.metrics.mark(
                request.request_id,
                "dit_enter",
                stage=diffusion_stage.value,
                instance_id=diffusion.instance_id,
            )
            self.stage_client.run_stage(diffusion_stage, request)
            self.metrics.mark(
                request.request_id,
                "dit_end",
                stage=diffusion_stage.value,
                instance_id=diffusion.instance_id,
            )
            self.metrics.mark(request.request_id, "vae_enter", stage=StageKind.VAE.value)
            self.stage_client.run_stage(StageKind.VAE, request)
            self.metrics.mark(request.request_id, "vae_end", stage=StageKind.VAE.value)
        self.metrics.mark(request.request_id, "request_done")
        return ShiftServeResponse(
            request_id=request.request_id,
            pe_instance=pe.instance_id,
            diffusion_instance=diffusion.instance_id,
            generated_tokens=generated,
            flip_direction=flip_direction,
        )

    def mark_draining(self, instance_ids: list[str]) -> None:
        for instance_id in instance_ids:
            if instance_id in self.instances:
                self.instances[instance_id].draining = True

    def activate(self, instance_ids: list[str]) -> None:
        for instance_id in instance_ids:
            if instance_id in self.instances:
                instance = self.instances[instance_id]
                instance.active = True
                instance.ready = True
                instance.draining = False
                instance.launching = False

    def _paired_te_instance_id(self, diffusion: InstanceRuntimeState) -> str | None:
        if diffusion.dit_vae_bundle is not None and diffusion.dit_vae_bundle.paired_te_id:
            return diffusion.dit_vae_bundle.paired_te_id
        same_node_te = sorted(
            (
                instance.instance_id
                for instance in self.instances.values()
                if instance.kind == StageKind.TE and instance.node_id == diffusion.node_id
            )
        )
        return same_node_te[0] if same_node_te else None

    @classmethod
    def for_dry_run(
        cls,
        *,
        out_dir: str,
        mode: SchedulerMode,
        short_bin: int = 512,
        long_bin: int = 2048,
    ) -> "ShiftServeRouter":
        instances = [
            InstanceRuntimeState("pe-0", StageKind.PE, "node-a"),
            InstanceRuntimeState("dit-0", StageKind.DIT, "node-b"),
        ]
        scheduler = ShiftServeScheduler(
            mode=mode,
            short_bin=short_bin,
            long_bin=long_bin,
        )
        return cls(
            instances=instances,
            scheduler=scheduler,
            metrics=MetricsRecorder(out_dir),
            flip_monitor=HysteresisFlipMonitor(short_bin=short_bin, long_bin=long_bin),
        )
