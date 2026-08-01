#!/bin/bash
# Launch torchspec.train_entry for Kimi-K3 MXFP4 Eagle3 3-node training (B200)
#
# Run this on the head node AFTER the Ray cluster is fully ready.
# See examples/kimi-k3-3node-b200/setup_ray_cluster.sh to set up the cluster first.
#
# Node layout:
#   Head node (this node): 8 GPUs (GPU 0-7) — FSDP draft training
#   Worker nodes x2:       8 GPUs each      — vLLM inference (TP=16)
#
# Usage:
#   bash examples/kimi-k3-3node-b200/run.sh [extra_overrides...]
#
# Environment variables:
#   TRAIN_GPUS          — GPUs for training (default: 8)
#   INFERENCE_GPUS      — Total inference GPUs across all worker nodes (default: 16)
#   INFERENCE_NODES     — Number of inference worker nodes (default: 2)
#   CONFIG_FILE         — Override config file path
#   NCCL_SOCKET_IFNAME  — network interface for NCCL/Gloo/TP (default: eth0)
#   MOONCAKE_DEVICE_NAME— RDMA NIC device names, comma-separated (e.g. mlx5_0).
#                         Find them with: ibdev2netdev -v
#   MOONCAKE_PROTOCOL   — rdma | tcp (default: leave to config; set tcp if no RDMA)

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCHSPEC_LOG_LEVEL=INFO

# Network interface for NCCL/Gloo/TP collectives — must match setup_ray_cluster.sh.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}

TRAIN_GPUS="${TRAIN_GPUS:-8}"
INFERENCE_GPUS="${INFERENCE_GPUS:-16}"
INFERENCE_NODES="${INFERENCE_NODES:-2}"

CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/configs/vllm_kimi_k3_3node.yaml}"

# Optional Mooncake overrides. If unset, the values baked into the YAML are used.
MOONCAKE_OVERRIDES=()
if [[ -n "${MOONCAKE_DEVICE_NAME:-}" ]]; then
  MOONCAKE_OVERRIDES+=("mooncake.device_name=$MOONCAKE_DEVICE_NAME")
fi
if [[ -n "${MOONCAKE_PROTOCOL:-}" ]]; then
  MOONCAKE_OVERRIDES+=("mooncake.protocol=$MOONCAKE_PROTOCOL")
fi

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/kimi_k3_3node_train_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

echo "=============================================="
echo "Kimi-K3 MXFP4 3-Node Training (B200)"
echo "=============================================="
echo "Config:                $CONFIG_FILE"
echo "  Training GPUs:  $TRAIN_GPUS (this node x $TRAIN_GPUS GPUs)"
echo "  Inference GPUs: $INFERENCE_GPUS ($INFERENCE_NODES nodes x 8 GPUs, TP=$INFERENCE_GPUS)"
echo "  dist_init_addr: (auto-negotiated via Ray)"
echo "  IFNAME:         $NCCL_SOCKET_IFNAME"
if [[ ${#MOONCAKE_OVERRIDES[@]} -gt 0 ]]; then
  echo "  Mooncake:       ${MOONCAKE_OVERRIDES[*]}"
fi
echo "=============================================="

if ! ray status &>/dev/null; then
  echo "ERROR: Cannot connect to Ray cluster (is Ray running on this node?)"
  echo "Start the cluster first:"
  echo "  NODE_ROLE=head     bash examples/kimi-k3-3node-b200/setup_ray_cluster.sh"
  echo "  HEAD_IP=<node0_ip> NODE_ROLE=worker bash examples/kimi-k3-3node-b200/setup_ray_cluster.sh"
  exit 1
fi

echo "=== Launching training ==="
# Topology (nnodes=2, tp_size=16, aux layers, MXFP4 auto-detect) is already
# encoded in the YAML; we only surface the GPU/node counts as overridable knobs.
python3 -m torchspec.train_entry \
  --config "$CONFIG_FILE" \
  training.training_num_gpus_per_node="$TRAIN_GPUS" \
  inference.inference_engine_type="vllm" \
  inference.inference_num_gpus="$INFERENCE_GPUS" \
  inference.inference_num_gpus_per_engine="$INFERENCE_GPUS" \
  inference.inference_num_gpus_per_node=8 \
  inference.vllm.tp_size="$INFERENCE_GPUS" \
  inference.vllm.nnodes="$INFERENCE_NODES" \
  "${MOONCAKE_OVERRIDES[@]}" \
  "$@"

echo "=============================================="
echo "Training completed!"
echo "=============================================="
