# Kimi-K3 MXFP4 3-Node Training (B200)

Production 3-node setup for training an Eagle3 draft model for **Kimi-K3**
(`moonshotai/Kimi-K3`) — a 2.8T-param (104B active) multimodal MoE with a
KDA/MLA hybrid text tower. The shipped checkpoint is already MXFP4-packed; vLLM
auto-detects the compressed-tensors format from the checkpoint.

The cluster/env plumbing reuses the verified
[`kimi-k3-2node-inference-b200`](../kimi-k3-2node-inference-b200) recipe
(shared `ray_env.sh` sourced before `ray start`, NCCL over TCP on the control
NIC, `VLLM_USE_DEEP_GEMM=0`, default `/tmp/ray` temp dir so `RAY_ADDRESS=auto`
works), plus the training-only pieces: Mooncake hidden-state transfer and
IP-pinned node roles.

> **Bring-up status & debugging record:** see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
> for every issue hit so far (root causes + fixes), what remains unverified,
> and the ranked structural fixes for a smooth training run.

## Node layout

Roles are pinned by IP via `training.placement_strategy=custom` — they do
**not** depend on Ray join order:

| Node | Role | GPUs |
|------|------|------|
| `10.1.33.20` | vLLM inference, node_rank 0 (TP dist-init master) | 8 |
| `10.1.33.22` | vLLM inference, node_rank 1 | 8 |
| `10.1.33.24` | Ray head + FSDP draft training | 8 |

The two inference nodes form one TP=16 vLLM engine (TorchSpec's multi-node
vLLM path computes `tp_size = nnodes × gpus_per_node`; there is no PP option
here, unlike the standalone 2-node inference example). Cross-node NCCL runs
over TCP on the control NIC; hidden states flow to the training node via
Mooncake (TCP by default).

## Prerequisites

- The three nodes above, 8× B200 each, sharing the `10.1.33.0/24` control
  subnet (auto-detected in `ray_env.sh`; override `SUBNET_PREFIX`/`NET_IF_CTRL`).
- **Python env: the repo `venv`** (default in `ray_env.sh`). It has torchspec
  (editable), `mooncake-transfer-engine`, and vLLM nightly with
  `KimiK3ForConditionalGeneration`. The shared `vllm_kimi_k3` conda env used by
  the 2-node inference example is *not* sufficient — it lacks
  omegaconf/mooncake. Because the venv lives on shared scratch, all nodes get
  identical python/torch/NCCL automatically.
- Model weights at `/sms-scratch/vamsik/checkpoints/Kimi-K3` (~1.45 TB MXFP4,
  `trust_remote_code=true`). First load from cold NFS takes hours; page-cached
  reruns take minutes.
- A training dataset **with completions** — see [Dataset](#dataset) below.
- Note: these are the same GPUs the 2-node inference example uses —
  `setup_ray_cluster.sh` tears down any existing Ray cluster on each node.

## How to run

### 1. Start the Ray cluster

On the **training node** (`10.1.33.24`):

```bash
NODE_ROLE=head bash examples/kimi-k3-3node-b200/setup_ray_cluster.sh
```

On **each inference node** (`10.1.33.20`, then `10.1.33.22`):

```bash
NODE_ROLE=worker HEAD_IP=10.1.33.24 \
  bash examples/kimi-k3-3node-b200/setup_ray_cluster.sh
```

Each node prints a preflight block (python/ray paths, torch/NCCL/CUDA/vLLM
versions, mooncake import) — the blocks must match line-for-line across nodes.

Verify from the head node:

```bash
ray status    # expect 3 nodes / 24 GPUs
```

### 2. Launch training

On the **training node** (`10.1.33.24`):

```bash
DATA_PATH=/path/to/train_conversations.jsonl \
  bash examples/kimi-k3-3node-b200/run.sh
```

## Dataset

**The data file needs both the prompt *and* the completion.** Training never
generates text: the target model runs a *prefill-only* forward pass
(`max_tokens=1`) over each full conversation to extract aux hidden states, and
the draft is trained on the assistant spans via the loss mask. Conversations
whose assistant spans are empty are dropped during preprocessing. (The vLLM
backend also doesn't support `train_with_decode`, so there is no
generate-during-training fallback.)

Format — one JSON object per line, assistant turns present:

```json
{"id": "...", "conversations": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "reasoning_content": "...", "content": "..."}]}
```

For the `kimi-k3` chat template, assistant reasoning goes in
`reasoning_content` (or an inline `<think>...</think>` block in `content`);
the parser renders it into K3's native think/response channels.

Any assistant text works mechanically (the target model is teacher-forced over
it), but **Kimi-K3's own completions are the right choice** — the draft should
learn the distribution it will speculate for, including the `<think>` channel.
Generate them with the 2-node inference example, then convert:

```bash
# 1) On the 2-node inference cluster: prompts -> K3 completions
DATA_PATH=examples/data/sample_conversations.jsonl \
  bash examples/kimi-k3-2node-inference-b200/run.sh

# 2) Convert completions JSONL -> training conversations JSONL
python examples/kimi-k3-3node-b200/completions_to_train_data.py \
  --input outputs/kimi_k3_2node_inference/completions_<ts>.jsonl \
  --output examples/data/kimi_k3_train_conversations.jsonl
```

The converter splits each raw completion at the K3 channel markers into
`reasoning_content` / `content` and drops truncated (`finish_reason != stop`)
completions by default.

## Common customizations

```bash
# Different node assignment (first inference IP becomes vLLM node_rank 0)
TRAIN_NODE_IP=10.1.33.24 INFER_NODE_IPS=10.1.33.20,10.1.33.22 \
  bash examples/kimi-k3-3node-b200/run.sh

# RDMA transport for Mooncake (only with a verified shared fabric!)
MOONCAKE_PROTOCOL=rdma MOONCAKE_DEVICE_NAME=mlx5_2,mlx5_5 \
  bash examples/kimi-k3-3node-b200/run.sh

# Degraded node: register fewer GPUs on the affected node, keep counts in sync
#   On that node:  NUM_GPUS=6 NODE_ROLE=... bash .../setup_ray_cluster.sh
#   On the head:   INFERENCE_GPUS=12 bash .../run.sh   # TP must divide 96

# Extra config overrides pass through to train_entry
bash examples/kimi-k3-3node-b200/run.sh training.learning_rate=1e-5
```

## Scripts

| Script | Purpose |
|--------|---------|
| `ray_env.sh` | Shared env (venv activate, NIC/NCCL config, vLLM knobs, Mooncake cudart) — sourced by both scripts, on all nodes |
| `setup_ray_cluster.sh` | Preflight + clean stale state + start Ray head/worker |
| `run.sh` | Launch training with IP-pinned roles (run on the training node) |
| `completions_to_train_data.py` | Convert 2-node inference completions into training conversations |

## Notes

- **`RAY_ADDRESS=auto` is required** and set by `run.sh`; discovery reads
  `/tmp/ray/session_latest`, which is why `setup_ray_cluster.sh` uses Ray's
  default temp dir (the old custom `--temp-dir` broke discovery and made
  vLLM's EngineCore start a local single-node Ray instance).
- **`ray_env.sh` must be sourced before `ray start`** — the raylet snapshots
  its environment (NCCL socket config, `VLLM_USE_DEEP_GEMM=0`, Mooncake's
  `libcudart.so.12` on `LD_LIBRARY_PATH`) and hands it to every actor.
- **Engine support (BLOCKER, unverified).** TorchSpec's `VllmEngine` +
  `MooncakeHiddenStatesConnector` path is unverified for the custom
  `KimiK3ForConditionalGeneration` (KDA/MLA hybrid) aux hidden-state
  extraction. Smoke-test with a handful of samples
  (`training.num_train_steps=3`) before committing to the full run.
- **Memory footprint.** TP=16 across 2 nodes assumes the native MXFP4 target
  fits with KV/activation headroom at `mem_fraction_static=0.85`; re-verify
  for your build.
- **Cross-node TP over TCP is slow.** The 2-node inference example gets away
  with TCP because its TP=8×PP=2 layout keeps TP all-reduces on NVLink; this
  training path has no PP option, so every layer's TP all-reduce crosses the
  TCP link (measured ~1.8 Gb/s on this cluster) — expect very slow prefill.
  Correctness first; for throughput, validate a rail-pinned RoCE
  `NCCL_IB_HCA` pairing for `10.1.33.20↔10.1.33.22` (the system-order-LCS
  procedure that hit ~60 GB/s on other node pairs; note .22's rail 43 /
  `mlx5_13` is dead) and relax the `NCCL_NET=Socket` block in `ray_env.sh`.
  That path also needs the raylet memlock fix
  (`sudo prlimit --memlock=unlimited:unlimited --pid $(pgrep -x raylet)`
  after every `ray start`).
