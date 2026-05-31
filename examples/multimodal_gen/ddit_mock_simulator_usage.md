# DDiT Mock Simulator Usage

This simulator is a CPU-only discrete-event model for DDiT scheduling experiments.
It models two nodes by default, with 8 GPUs per node, and keeps each request's
text encoder, DiT, and VAE phases on one node.

## Single Case

```bash
python3 examples/multimodal_gen/ddit_mock_simulator.py \
  --profile-path examples/multimodal_gen/ddit_profile_a100_data.json \
  --model-id z-image \
  --num-nodes 2 --gpus-per-node 8 \
  --policy hungry_first \
  --window-size 8 \
  --resolutions 720p,2k --ratios 0.75,0.25 \
  --num-requests 128 \
  --rate 1.0 \
  --seed 42 \
  --out-dir /data/outputs/ddit_mock_sims/case_hungry_rr1
```

Outputs:

- `ddit_lifecycle.csv`: one row per request.
- `ddit_rank_switch.jsonl`: DiT start, DiT scale-up, and DiT-to-VAE rank changes.
- `ddit_op_trace.jsonl`: text encoder, DiT step, and VAE timing events.
- `ddit_lifecycle_summary.csv`: one-row case summary.
- `summary.json`: same summary in JSON.

## Sweep

```bash
python3 examples/multimodal_gen/run_ddit_mock_sweep.py \
  --profile-path examples/multimodal_gen/ddit_profile_a100_data.json \
  --model-id z-image \
  --num-nodes 2 --gpus-per-node 8 \
  --policies naive,naive_greedy,hungry_first,wsjf,wsjf_scale_up \
  --window-size 8 \
  --resolutions 720p,2k --ratios 0.75,0.25 \
  --num-requests 128 \
  --rates 0.2,0.6,1.0,2.0,burst \
  --seed 42 \
  --out-dir /data/outputs/ddit_mock_sims/zimage_a100_2nodes
```

Each case is written to:

```text
<out-dir>/rr_<rate>/<policy>/
```

The top-level summary is `summary.xlsx` when `openpyxl` is installed. Otherwise,
the script writes one CSV per metric, such as `summary_p50.csv`.

## Linux nohup

```bash
mkdir -p /data/outputs/ddit_mock_sims/zimage_a100_2nodes
nohup python3 examples/multimodal_gen/run_ddit_mock_sweep.py \
  --profile-path examples/multimodal_gen/ddit_profile_a100_data.json \
  --model-id z-image \
  --num-nodes 2 --gpus-per-node 8 \
  --policies naive,naive_greedy,hungry_first,wsjf,wsjf_scale_up \
  --window-size 8 \
  --resolutions 720p,2k --ratios 0.75,0.25 \
  --num-requests 128 \
  --rates 0.2,0.6,1.0,2.0,burst \
  --seed 42 \
  --out-dir /data/outputs/ddit_mock_sims/zimage_a100_2nodes \
  > /data/outputs/ddit_mock_sims/zimage_a100_2nodes/nohup.log 2>&1 &
```

## Metrics

- `p50`, `p90`, `p99`, `mean`, `max`: request latency from `add_time` to `vae_end`.
- `slo5`, `slo10`: SLO attainment. Unit SLO is
  `text_encoder_time + vae_time[8] + dit_step_time[8] * dit_step_num`, computed
  per resolution, then multiplied by 5 or 10.
- `throughput`: completed requests divided by makespan.
- `makespan`: first arrival to last VAE completion.
- `completed_count`: must equal `--num-requests`; otherwise the simulator exits
  with an error.

## Policy Notes

- `naive`: FIFO, fixed opt DiT ranks, VAE uses final DiT ranks.
- `naive_greedy`: FIFO, starts with the largest available allowed rank count not
  exceeding opt, VAE uses final DiT ranks.
- `hungry_first`: FIFO admission, repeatedly scales up the globally highest
  starvation-score running request while any node-local free ranks remain, VAE
  uses profile `opt_vae_k` or 1.
- `wsjf`: windowed shortest estimated remaining DiT time, window size defaults to 8.
- `wsjf_scale_up`: hungry-style scale-up first, then WSJF windowed start.

`--te-resource-mode lane` is the default and models one node-local text encoder
lane that does not consume DiT/VAE ranks. Use `--te-resource-mode exclusive` if
you want text encoder to occupy the full node conservatively.
