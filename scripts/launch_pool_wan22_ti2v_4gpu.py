"""
Launch a single-machine pooled diffusion deployment for Wan2.2 TI2V 5B.

This script uses the Python API `launch_pool_disagg_server`, which is the
"real" pool mode in SGLang diffusion disaggregation. In this mode:

1. We do NOT manually start `sglang serve --disagg-role server`.
2. We do NOT manually start `encoder` / `denoiser` / `decoder` role processes.
3. The Python launcher spawns:
   - 1 head HTTP server
   - 1 encoder instance
   - 1 denoiser instance
   - 1 decoder instance
4. Endpoints for work / control / result are derived automatically from the
   head scheduler port.

Requested topology in this file:

- encoder count   = 1
- denoiser count  = 1
- decoder count   = 1
- encoder TP      = 4
- denoiser Ulysses/SP = 4
- decoder SP      = 4
- all three roles use GPUs [4, 5, 6, 7]

This means:
- encoder instance uses GPUs [4, 5, 6, 7] with TP=4
- denoiser instance uses GPUs [4, 5, 6, 7] with SP/Ulysses=4
- decoder instance uses GPUs [4, 5, 6, 7] with SP=4

The same GPU group is reused across the three role types. This is supported
by pool mode because roles run as separate instances coordinated by the head
DiffusionServer.

Important debugging choice:
- Offload is disabled here on purpose so that the roles visibly occupy GPU
  memory after startup and while serving requests.

How to run:

    python scripts/launch_pool_wan22_ti2v_4gpu.py

How to send a TI2V request after startup:

1. Multipart upload, safest:

    curl -X POST http://127.0.0.1:30010/v1/videos ^
      -F "model=Wan2.2-TI2V-5B-Diffusers" ^
      -F "prompt=Turn the man upside down" ^
      -F "size=1280x704" ^
      -F "seconds=4" ^
      -F "num_inference_steps=30" ^
      -F "input_reference=@/absolute/path/to/example_image.png"

2. JSON with a server-local image path:

    curl -X POST http://127.0.0.1:30010/v1/videos ^
      -H "Content-Type: application/json" ^
      -d "{\"model\":\"Wan2.2-TI2V-5B-Diffusers\",\"prompt\":\"Turn the man upside down\",\"size\":\"1280x704\",\"seconds\":\"4\",\"num_inference_steps\":30,\"input_reference\":\"/absolute/path/on/server/example_image.png\"}"

3. Poll job status:

    curl http://127.0.0.1:30010/v1/videos/<video_id>

4. Download result:

    curl -L -o out.mp4 http://127.0.0.1:30010/v1/videos/<video_id>/content

Port meaning in this script:

- HTTP_PORT = 30010
  The public HTTP/OpenAI-compatible API. Your curl requests go here.

- HEAD_SCHEDULER_PORT = 30020
  The head DiffusionServer frontend. Role instances register to this port and
  head dispatches jobs through it. This is internal ZMQ traffic, not HTTP.

- HEAD_SCHEDULER_PORT + 1 = 30021
  Internal result endpoint for encoder -> head.

- HEAD_SCHEDULER_PORT + 2 = 30022
  Internal result endpoint for denoiser -> head.

- HEAD_SCHEDULER_PORT + 3 = 30023
  Internal result endpoint for decoder -> head.

What you should see in logs:

- head side:
  `DiffusionServer transfer: registered encoder[0] ...`
  `DiffusionServer transfer: registered denoiser[0] ...`
  `DiffusionServer transfer: registered decoder[0] ...`

- role side:
  `Transfer ENCODER: registered with DS`
  `Transfer DENOISER: registered with DS`
  `Transfer DECODER: registered with DS`

If requests are accepted but stay queued, look for missing registration logs.
"""

from __future__ import annotations

from sglang.multimodal_gen.runtime.launch_server import launch_pool_disagg_server
from sglang.multimodal_gen.runtime.server_args import ServerArgs


# ----------------------------
# User-editable deployment cfg
# ----------------------------
MODEL_PATH = "/data/Wan2_2_TI2V_5B"
MODEL_ID = "Wan2.2-TI2V-5B-Diffusers"

HOST = "127.0.0.1"
HTTP_PORT = 30010
HEAD_SCHEDULER_PORT = 30020

# One encoder instance, one denoiser instance, one decoder instance.
# Each instance uses the same 4 GPUs: [4, 5, 6, 7].
GPU_GROUP = [4, 5, 6, 7]

# Debug-friendly defaults:
# - Disable offload so GPU memory usage is visible.
# - Use max_slots_per_instance=1 to simplify early bring-up.
LOG_LEVEL = "debug"


def build_server_args() -> ServerArgs:
    """Construct ServerArgs for pooled 1/1/1 role deployment."""
    return ServerArgs.from_kwargs(
        model_path=MODEL_PATH,
        model_id=MODEL_ID,
        host=HOST,
        port=HTTP_PORT,
        scheduler_port=HEAD_SCHEDULER_PORT,
        log_level=LOG_LEVEL,
        # Requested role parallelism.
        encoder_tp=4,
        denoiser_sp=4,
        denoiser_ulysses=4,
        denoiser_ring=1,
        decoder_sp=4,
        # Make sure roles run on GPU and keep modules on GPU.
        disagg_role_device="cuda",
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        dit_layerwise_offload=False,
        dit_cpu_offload=False,
        # Helpful for debugging and simpler scheduling.
        disagg_dispatch_policy="round_robin",
        disagg_max_slots_per_instance=1,
        disagg_timeout=3600,
        disagg_downstream_wait_timeout=1800,
        warmup=True,
    )


def main() -> None:
    server_args = build_server_args()

    print("Launching pooled disaggregated diffusion server...")
    print(f"  model_path       : {MODEL_PATH}")
    print(f"  model_id         : {MODEL_ID}")
    print(f"  host/http_port   : {HOST}:{HTTP_PORT}")
    print(f"  head_sched_port  : {HEAD_SCHEDULER_PORT}")
    print(f"  encoder_gpus     : {[GPU_GROUP]} (tp=4)")
    print(f"  denoiser_gpus    : {[GPU_GROUP]} (sp=4, ulysses=4, ring=1)")
    print(f"  decoder_gpus     : {[GPU_GROUP]} (sp=4)")
    print("")
    print("Expected internal head endpoints:")
    print(f"  frontend         : tcp://{HOST}:{HEAD_SCHEDULER_PORT}")
    print(f"  encoder_result   : tcp://{HOST}:{HEAD_SCHEDULER_PORT + 1}")
    print(f"  denoiser_result  : tcp://{HOST}:{HEAD_SCHEDULER_PORT + 2}")
    print(f"  decoder_result   : tcp://{HOST}:{HEAD_SCHEDULER_PORT + 3}")
    print("")
    print("HTTP API:")
    print(f"  health           : http://{HOST}:{HTTP_PORT}/health")
    print(f"  create video     : http://{HOST}:{HTTP_PORT}/v1/videos")
    print("")
    print("Waiting for role registration logs before sending requests...")

    launch_pool_disagg_server(
        server_args,
        encoder_gpus=[GPU_GROUP],
        denoiser_gpus=[GPU_GROUP],
        decoder_gpus=[GPU_GROUP],
    )


if __name__ == "__main__":
    main()
