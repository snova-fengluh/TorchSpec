#!/bin/bash
# Launch Kimi-K3 2-node batch inference (B200) with vLLM over Ray.
#
# Run this on the head node AFTER the Ray cluster is fully ready
# (see setup_ray_cluster.sh; `ray status` should show 2 nodes / 16 GPUs).
#
# Node layout (TP=8 x PP=2 by default — TP all-reduces stay on intra-node
# NVLink; only pipeline-stage activations cross the inter-node fabric):
#   Head node (this node): 8 GPUs — vLLM driver + pipeline stage 0 (TP=8)
#   Worker node:           8 GPUs — pipeline stage 1 (TP=8)
#
# Usage:
#   bash examples/kimi-k3-2node-inference-b200/run.sh [extra generate.py flags...]
#
# Environment variables:
#   MODEL_PATH          — Kimi-K3 checkpoint (default: /sms-scratch/vamsik/checkpoints/Kimi-K3)
#   DATA_PATH           — conversations JSONL (default: examples/data/sample_conversations.jsonl)
#   OUTPUT_PATH         — output JSONL (default: outputs/kimi_k3_2node_inference/completions_<ts>.jsonl)
#   TP_SIZE             — tensor parallel size (default: 8). Must divide Kimi-K3's
#                         96 attention heads (16, 12, 8, ...).
#   PP_SIZE             — pipeline parallel size (default: 2). TP_SIZE*PP_SIZE must
#                         equal the total GPUs registered with Ray.
#   RAY_ADDRESS         — Ray cluster address (default: auto). REQUIRED so vLLM's
#                         EngineCore joins the 2-node cluster instead of starting
#                         a local single-node Ray instance.
#   MAX_NEW_TOKENS      — max completion length (default: 4096)
#   TEMPERATURE         — sampling temperature (default: 0.7)
#   ENFORCE_EAGER       — 1 (default) disables torch.compile — the verified
#                         config; set 0 for peak throughput once stable
#   NET_IF_CTRL         — cross-node NIC; auto-detected in ray_env.sh

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Same env the raylets were started under (conda env with vLLM nightly,
# NCCL socket config, VLLM_USE_DEEP_GEMM=0, unset VLLM_PORT, ...).
source "$SCRIPT_DIR/ray_env.sh"

# Make vLLM join the existing 2-node Ray cluster. Without this, vLLM's
# EngineCore subprocess calls ray.init() with no address and starts a NEW
# local Ray instance that only sees this node's 8 GPUs ("Started a local Ray
# instance" + placement group allocation failures in the logs).
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"

MODEL_PATH="${MODEL_PATH:-/dev/shm/Kimi-K3}"
# DATA_PATH="${DATA_PATH:-$ROOT_DIR/examples/data/sample_conversations.jsonl}"
DATA_PATH="${DATA_PATH:-/sms-scratch/ravira/datasets/train_regen_code_math_merged.jsonl}"
TP_SIZE="${TP_SIZE:-8}"
PP_SIZE="${PP_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
EAGER_FLAG=()
[[ "$ENFORCE_EAGER" == "1" ]] && EAGER_FLAG=(--enforce-eager)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/outputs/kimi_k3_2node_inference/train_regen_code_math_merged_${TIMESTAMP}.jsonl}"

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/kimi_k3_2node_inference_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

echo "=============================================="
echo "Kimi-K3 2-Node Batch Inference (B200)"
echo "=============================================="
echo "  Model:          $MODEL_PATH"
echo "  Dataset:        $DATA_PATH"
echo "  Output:         $OUTPUT_PATH"
echo "  Parallelism:    TP=$TP_SIZE x PP=$PP_SIZE ($((TP_SIZE * PP_SIZE)) GPUs, 2 nodes via Ray)"
echo "  RAY_ADDRESS:    $RAY_ADDRESS"
echo "  Max new tokens: $MAX_NEW_TOKENS  Temperature: $TEMPERATURE"
echo "  Enforce eager:  $ENFORCE_EAGER"
echo "  IFNAME:         $NCCL_SOCKET_IFNAME"
echo "=============================================="

if ! ray status &>/dev/null; then
  echo "ERROR: Cannot connect to Ray cluster (is Ray running on this node?)"
  echo "Start the cluster first:"
  echo "  NODE_ROLE=head     bash examples/kimi-k3-2node-inference-b200/setup_ray_cluster.sh"
  echo "  HEAD_IP=<node0_ip> NODE_ROLE=worker bash examples/kimi-k3-2node-inference-b200/setup_ray_cluster.sh"
  exit 1
fi

echo "=== Launching batch inference ==="
python3 "$SCRIPT_DIR/generate.py" \
  --model "$MODEL_PATH" \
  --input "$DATA_PATH" \
  --output "$OUTPUT_PATH" \
  --tp "$TP_SIZE" \
  --pp "$PP_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  "${EAGER_FLAG[@]}" \
  "$@"

echo "=============================================="
echo "Inference completed! Output: $OUTPUT_PATH"
echo "=============================================="
