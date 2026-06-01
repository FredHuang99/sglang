"""Launch command construction and port validation for ShiftServe."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Iterable

from sglang.shiftserve.config import DeploymentConfig, InstanceConfig

TWO_GIB = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class LaunchDefaults:
    transfer_pool_size: int = TWO_GIB
    max_slots_per_instance: int = 1
    transfer_pin_memory: str = "auto"
    disagg_timeout: int = 3600
    disagg_downstream_timeout: int = 1800
    pe_chunked_prefill_size: int = 512
    pe_max_total_tokens: int = 4096
    pe_cuda_graph_max_bs: int = 1
    pe_max_running_requests: int = 1
    pe_mem_fraction_static: float | None = None
    rank0_broadcast: bool = False


class PortAllocator:
    """Validates explicit ports before a launch mutates the machine."""

    REQUIRED_PORT_NAMES = {
        "pe": {"http"},
        "te": {"work", "control"},
        "dit": {"work", "control"},
        "vae": {"work", "control"},
        "dit_vae": {"work", "control"},
    }

    @staticmethod
    def validate(deployment: DeploymentConfig) -> None:
        seen: dict[tuple[str, int], str] = {}
        for instance in deployment.instances.values():
            required = PortAllocator.REQUIRED_PORT_NAMES[instance.kind]
            missing = sorted(required - set(instance.ports))
            if missing:
                raise ValueError(
                    f"Instance {instance.id} is missing required ports: {missing}"
                )
            for name, port in instance.ports.items():
                if not 1 <= int(port) <= 65535:
                    raise ValueError(
                        f"Instance {instance.id} port {name}={port} is outside 1..65535"
                    )
                node_port = (instance.node_id, int(port))
                if node_port in seen:
                    raise ValueError(
                        f"Port collision on node {instance.node_id}:{port}: "
                        f"{seen[node_port]} and {instance.id}.{name}"
                    )
                seen[node_port] = f"{instance.id}.{name}"


class LaunchCommandBuilder:
    """Builds argv lists for PE and diffusion role processes."""

    def __init__(self, defaults: LaunchDefaults | None = None):
        self.defaults = defaults or LaunchDefaults()

    def build_pe_command(
        self,
        instance: InstanceConfig,
        *,
        model_path: str,
        host: str,
        trust_remote_code: bool = True,
    ) -> list[str]:
        if instance.kind != "pe":
            raise ValueError(f"build_pe_command expects kind=pe, got {instance.kind}")
        args = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            host,
            "--port",
            str(instance.ports["http"]),
            "--tp-size",
            str(max(instance.ranks, 1)),
            "--disable-piecewise-cuda-graph",
            "--cuda-graph-max-bs",
            str(self.defaults.pe_cuda_graph_max_bs),
            "--chunked-prefill-size",
            str(self.defaults.pe_chunked_prefill_size),
            "--max-running-requests",
            str(self.defaults.pe_max_running_requests),
            "--max-total-tokens",
            str(self.defaults.pe_max_total_tokens),
            "--skip-server-warmup",
        ]
        if trust_remote_code:
            args.append("--trust-remote-code")
        if self.defaults.pe_mem_fraction_static is not None:
            args.extend(["--mem-fraction-static", str(self.defaults.pe_mem_fraction_static)])
        if instance.gpu_ids:
            args.extend(["--base-gpu-id", str(min(instance.gpu_ids))])
        return args

    def build_diffusion_role_command(
        self,
        instance: InstanceConfig,
        *,
        model_path: str,
        server_addr: str,
        host: str,
        model_id: str | None = None,
        weighted_schedule: bool = False,
    ) -> list[str]:
        if instance.kind not in {"te", "dit", "vae", "dit_vae"}:
            raise ValueError(
                f"build_diffusion_role_command expects diffusion kind, got {instance.kind}"
            )
        role = {
            "te": "encoder",
            "dit": "denoiser",
            "vae": "decoder",
            "dit_vae": "dit_vae",
        }[instance.kind]
        args = [
            "python",
            "-m",
            "sglang.multimodal_gen.runtime.entrypoints.cli.main",
            "serve",
            "--model-path",
            model_path,
            "--host",
            host,
            "--scheduler-port",
            str(instance.ports["work"]),
            "--num-gpus",
            str(len(instance.gpu_ids) if instance.gpu_ids else max(instance.ranks, 1)),
            "--disagg-role",
            role,
            "--disagg-server-addr",
            server_addr,
            "--disagg-dispatch-policy",
            "weighted_shiftserve" if weighted_schedule else "round_robin",
            "--disagg-transfer-pool-size",
            str(self.defaults.transfer_pool_size),
            "--disagg-transfer-calibration-mode",
            "fixed",
            "--disagg-max-slots-per-instance",
            str(self.defaults.max_slots_per_instance),
            "--disagg-transfer-pin-memory",
            self.defaults.transfer_pin_memory,
            "--disagg-timeout",
            str(self.defaults.disagg_timeout),
            "--disagg-downstream-wait-timeout",
            str(self.defaults.disagg_downstream_timeout),
            "--warmup",
            "false",
            "--dit-cpu-offload",
            "false",
            "--dit-layerwise-offload",
            "false",
            "--text-encoder-cpu-offload",
            "false",
            "--image-encoder-cpu-offload",
            "false",
            "--vae-cpu-offload",
            "false",
            "--pin-cpu-memory",
            "false",
        ]
        if model_id or instance.model_id:
            args.extend(["--model-id", model_id or instance.model_id or ""])
        if instance.gpu_ids:
            args.extend(["--gpu-ids", ",".join(str(v) for v in instance.gpu_ids)])
        if instance.kind == "te":
            args.extend(["--disagg-role-device", instance.device])
        if instance.kind in {"dit", "dit_vae"}:
            args.extend(["--sp-degree", str(max(instance.ranks, 1))])
        if self.defaults.rank0_broadcast and instance.kind in {"dit", "vae", "dit_vae"}:
            args.extend(
                [
                    "--diffusion-weight-load-mode",
                    "rank0-broadcast",
                    "--diffusion-weight-broadcast-components",
                    "transformer,vae",
                ]
            )
        return args

    def build_launch_plan(
        self,
        deployment: DeploymentConfig,
        *,
        pe_model_path: str,
        diffusion_model_path: str,
        server_addr: str,
        weighted_schedule: bool = False,
    ) -> dict[str, list[str]]:
        PortAllocator.validate(deployment)
        commands: dict[str, list[str]] = {}
        for instance in deployment.instances.values():
            node = deployment.nodes[instance.node_id]
            host = node.host
            if instance.kind == "pe":
                commands[instance.id] = self.build_pe_command(
                    instance,
                    model_path=pe_model_path,
                    host=host,
                )
            else:
                commands[instance.id] = self.build_diffusion_role_command(
                    instance,
                    model_path=diffusion_model_path,
                    server_addr=server_addr,
                    host=host,
                    weighted_schedule=weighted_schedule,
                )
        return commands


def format_command(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def iter_launch_table(commands: dict[str, list[str]]) -> Iterable[tuple[str, str]]:
    for key in sorted(commands):
        yield key, format_command(commands[key])


def flatten_ports(deployment: DeploymentConfig) -> list[tuple[str, int, str]]:
    rows = []
    for instance in deployment.instances.values():
        rows.extend(
            (instance.node_id, int(port), f"{instance.id}.{name}")
            for name, port in instance.ports.items()
        )
    return sorted(rows, key=lambda row: (row[0], row[1], row[2]))


def has_duplicate_ports(deployment: DeploymentConfig) -> bool:
    ports = [(node, port) for node, port, _ in flatten_ports(deployment)]
    return len(set(ports)) != len(ports)
