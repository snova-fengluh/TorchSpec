#!/bin/bash
# Launch torchspec.train_entry for Kimi-K3 MXFP4 Eagle3 3-node training (B200)
#
# Run this on the Ray HEAD node (the training node, 10.1.33.24) AFTER the
# cluster is fully ready (see setup_ray_cluster.sh; `ray status` should show
# 3 nodes / 24 GPUs).
#
# Node layout — pinned by IP via training.placement_strategy=custom, so it does
# NOT depend on Ray join order:
#   10.1.33.20 — vLLM inference node_rank 0 (dist-init master for the TP group)
#   10.1.33.22 — vLLM inference node_rank 1
#   10.1.33.24 — FSDP draft training (this node, Ray head)
#
# Usage:
#   bash examples/kimi-k3-3node-b200/run.sh [extra_overrides...]
#
# Environment variables:
#   TRAIN_NODE_IP       — training node IP (default: 10.1.33.24)
#   INFER_NODE_IPS      — comma-separated inference node IPs, ORDER MATTERS:
#                         the first entry becomes vLLM node_rank 0
#                         (default: 10.1.33.20,10.1.33.22)
#   TRAIN_GPUS          — GPUs for training (default: 8)
#   INFERENCE_GPUS      — total inference GPUs across worker nodes (default: 16)
#   INFERENCE_NODES     — number of inference nodes (default: 2)
#   DATA_PATH           — training conversations JSONL. Must contain assistant
#                         completions (the target model runs prefill-only over
#                         them to extract hidden states — it does NOT generate).
#   EVAL_DATA_PATH      — eval conversations JSONL
#   CONFIG_FILE         — override config file path
#   MOONCAKE_PROTOCOL   — tcp (default) | rdma. This cluster has no shared
#                         RDMA fabric across arbitrary node trios; tcp is the
#                         safe verified transport.
#   MOONCAKE_DEVICE_NAME— RDMA NIC device names, comma-separated (rdma only).
#                         Find them with: ibdev2netdev -v
#   RAY_ADDRESS         — Ray cluster address (default: auto)

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Same env the raylets were started under (venv with torchspec+mooncake+vLLM
# nightly, NCCL socket config, VLLM_USE_DEEP_GEMM=0, unset VLLM_PORT, ...).
source "$SCRIPT_DIR/ray_env.sh"

# Join the existing 3-node Ray cluster. Without this, vLLM's EngineCore
# subprocess starts a NEW local single-node Ray instance ("Started a local Ray
# instance" + placement group allocation failures in the logs).
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"

# --- node-role pinning (custom placement) ---
TRAIN_NODE_IP="${TRAIN_NODE_IP:-10.1.33.20}"
INFER_NODE_IPS="${INFER_NODE_IPS:-10.1.33.26,10.1.33.12}"

TRAIN_GPUS="${TRAIN_GPUS:-8}"
INFERENCE_GPUS="${INFERENCE_GPUS:-16}"
INFERENCE_NODES="${INFERENCE_NODES:-2}"
# GPUs used per inference node. Must equal INFERENCE_GPUS / INFERENCE_NODES and
# match the --num-gpus each worker registered with Ray. TP size (= INFERENCE_GPUS,
# computed by the engine as nnodes * gpus_per_node) must divide Kimi-K3's
# 96 attention heads (16, 12, 8, ...).
INFERENCE_GPUS_PER_NODE="${INFERENCE_GPUS_PER_NODE:-$((INFERENCE_GPUS / INFERENCE_NODES))}"

CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/configs/vllm_kimi_k3_3node.yaml}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/outputs/kimi_k3_2node_inference/train_regen_code_math_merged_20260813_150025_converted_1000.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$ROOT_DIR/examples/data/eval_conversations.jsonl}"

# --- model paths (YAML values used when unset) ---
# TARGET_MODEL_PATH    — shared-FS checkpoint for driver/training actors
# INFERENCE_MODEL_PATH — node-local staged copy (e.g. /dev/shm/Kimi-K3) that
#                        ONLY the inference engines load from; must exist on
#                        every inference node
MODEL_OVERRIDES=()
if [[ -n "${TARGET_MODEL_PATH:-}" ]]; then
  MODEL_OVERRIDES+=("model.target_model_path=$TARGET_MODEL_PATH")
fi
if [[ -n "${INFERENCE_MODEL_PATH:-}" ]]; then
  MODEL_OVERRIDES+=("model.inference_model_path=$INFERENCE_MODEL_PATH")
fi

# --- Mooncake transport ---
# Default to TCP: the YAML ships protocol=rdma with placeholder NIC names, but
# this cluster's only verified cross-node transport is TCP on the control NIC.
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
MOONCAKE_OVERRIDES=("mooncake.protocol=$MOONCAKE_PROTOCOL")
if [[ -n "${MOONCAKE_DEVICE_NAME:-}" ]]; then
  MOONCAKE_OVERRIDES+=("mooncake.device_name=$MOONCAKE_DEVICE_NAME")
elif [[ "$MOONCAKE_PROTOCOL" == "tcp" ]]; then
  # Blank out the YAML's placeholder mlx5_* list — meaningless for TCP.
  MOONCAKE_OVERRIDES+=("mooncake.device_name=''")
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
echo "  Config:         $CONFIG_FILE"
echo "  Train data:     $DATA_PATH"
echo "  Training node:  $TRAIN_NODE_IP ($TRAIN_GPUS GPUs)"
echo "  Inference nodes: $INFER_NODE_IPS ($INFERENCE_NODES nodes x $INFERENCE_GPUS_PER_NODE GPUs, TP=$INFERENCE_GPUS)"
echo "  RAY_ADDRESS:    $RAY_ADDRESS"
echo "  Mooncake:       ${MOONCAKE_OVERRIDES[*]}"
echo "  IFNAME:         $NCCL_SOCKET_IFNAME"
echo "=============================================="

if ! ray status &>/dev/null; then
  echo "ERROR: Cannot connect to Ray cluster (is Ray running on this node?)"
  echo "Start the cluster first (see setup_ray_cluster.sh):"
  echo "  On $TRAIN_NODE_IP:            NODE_ROLE=head   bash examples/kimi-k3-3node-b200/setup_ray_cluster.sh"
  echo "  On ${INFER_NODE_IPS//,/ and }: NODE_ROLE=worker HEAD_IP=$TRAIN_NODE_IP bash examples/kimi-k3-3node-b200/setup_ray_cluster.sh"
  exit 1
fi

echo "=== Launching training ==="
# Topology (nnodes=2, tp_size=16, aux layers, MXFP4 auto-detect) is already
# encoded in the YAML; here we pin roles to physical nodes and surface the
# GPU/node counts as overridable knobs.
python3 -m torchspec.train_entry \
  --config "$CONFIG_FILE" \
  "dataset.train_data_path=$DATA_PATH" \
  "dataset.eval_data_path=$EVAL_DATA_PATH" \
  training.placement_strategy=custom \
  "training.training_node_ips=[$TRAIN_NODE_IP]" \
  "inference.inference_node_ips=[$INFER_NODE_IPS]" \
  training.training_num_gpus_per_node="$TRAIN_GPUS" \
  inference.inference_engine_type="vllm" \
  inference.inference_num_gpus="$INFERENCE_GPUS" \
  inference.inference_num_gpus_per_engine="$INFERENCE_GPUS" \
  inference.inference_num_gpus_per_node="$INFERENCE_GPUS_PER_NODE" \
  inference.vllm.tp_size="$INFERENCE_GPUS" \
  inference.vllm.nnodes="$INFERENCE_NODES" \
  "${MODEL_OVERRIDES[@]}" \
  "${MOONCAKE_OVERRIDES[@]}" \
  "$@"

echo "=============================================="
echo "Training completed!"
echo "=============================================="
