#!/bin/bash
# Ray cluster setup for Kimi-K3 2-node inference (B200)
#
# Node layout (2 nodes x 8 B200 = 16 GPUs):
#   Node 0 (head):   Ray head + vLLM driver, 8 GPUs
#   Node 1 (worker): vLLM workers, 8 GPUs
#
# Inference-only: the only cross-node traffic is NCCL over TCP on the control
# NIC (configured in ray_env.sh). No IB/RDMA/Mooncake required.
#
# Usage — run on each node with the appropriate NODE_ROLE:
#
#   # Node 0 (Ray head):
#   NODE_ROLE=head bash examples/kimi-k3-2node-inference-b200/setup_ray_cluster.sh
#
#   # Node 1 (worker):
#   NODE_ROLE=worker HEAD_IP=<node0_ip> \
#     bash examples/kimi-k3-2node-inference-b200/setup_ray_cluster.sh
#
# Environment variables:
#   NODE_ROLE   — "head" | "worker"
#   HEAD_IP     — control-NIC IP of node 0. Required only on the worker.
#   RAY_PORT    — Ray GCS port (default: 6379)
#   NUM_GPUS    — GPUs to register with Ray on this node (default: 8). Lower it
#                 on a node with a dead GPU; keep TP*PP == total registered GPUs.
#   NET_IF_CTRL — cross-node NIC; auto-detected in ray_env.sh, override if needed.

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Env (conda activate, NCCL_*, VLLM_*) MUST be in place before `ray start`:
# the raylet snapshots its environment and hands it to every vLLM worker.
source "$SCRIPT_DIR/ray_env.sh"

NODE_ROLE="${NODE_ROLE:?NODE_ROLE must be set to head or worker}"
RAY_PORT="${RAY_PORT:-6379}"
NUM_GPUS="${NUM_GPUS:-8}"

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/kimi_k3_2node_inf_ray_${NODE_ROLE}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

echo "=============================================="
echo "Kimi-K3 2-Node Inference Ray Cluster (B200)"
echo "NODE_ROLE: $NODE_ROLE  CTRL_IP: $LOCAL_CTRL_IP  IFACE: $NET_IF_CTRL"
echo "=============================================="

# Preflight (multi_node_ray_setup.md Step 2c): python/ray must resolve to the
# same env, and torch/NCCL versions must match across nodes — a mismatch makes
# NCCL init hang silently. Compare this block's output between the two nodes.
echo "=== Preflight (compare across nodes; all lines must match) ==="
echo "python: $(which python)"
echo "ray:    $(which ray)"
python - <<'EOF'
import sys, torch
print(f"exe   : {sys.executable}")
print(f"torch : {torch.__version__}")
print(f"nccl  : {torch.cuda.nccl.version()}")
print(f"cuda  : {torch.version.cuda}")
EOF

# Clean stale Ray state — a prior crashed run leaves ghost raylets and stale
# session files that confuse a fresh cluster. Scoped to this user.
echo "=== Cleaning stale Ray state ==="
ray stop --force 2>/dev/null || true
pkill -9 -u "$USER" -f 'ray::|raylet|gcs_server' 2>/dev/null || true
rm -rf /tmp/ray/session_* /tmp/ray/ray_current_cluster 2>/dev/null || true

# Default /tmp/ray temp dir on purpose: RAY_ADDRESS=auto discovers the cluster
# via /tmp/ray/session_latest — a custom --temp-dir breaks that discovery and
# makes vLLM's EngineCore start its own single-node Ray instance.
case "$NODE_ROLE" in
  head)
    echo "=== Starting Ray HEAD node ($NUM_GPUS GPUs) ==="
    ray start --head \
      --port "$RAY_PORT" \
      --node-ip-address "$LOCAL_CTRL_IP" \
      --num-gpus "$NUM_GPUS" \
      --disable-usage-stats
    echo "Ray head started at $LOCAL_CTRL_IP:$RAY_PORT"
    echo "Next steps:"
    echo "  Worker node: NODE_ROLE=worker HEAD_IP=$LOCAL_CTRL_IP bash examples/kimi-k3-2node-inference-b200/setup_ray_cluster.sh"
    echo "  Verify:      ray status   # expect 2 nodes / 16 GPUs"
    echo "  Inference:   bash examples/kimi-k3-2node-inference-b200/run.sh"
    ;;

  worker)
    HEAD_IP="${HEAD_IP:?HEAD_IP must be set to node 0\'s control-NIC IP}"
    echo "=== Joining Ray cluster at ${HEAD_IP}:${RAY_PORT} as WORKER ($NUM_GPUS GPUs) ==="
    ray start \
      --address "${HEAD_IP}:${RAY_PORT}" \
      --node-ip-address "$LOCAL_CTRL_IP" \
      --num-gpus "$NUM_GPUS" \
      --disable-usage-stats
    echo "Worker joined Ray cluster at ${HEAD_IP}:${RAY_PORT}"
    ;;

  *)
    echo "ERROR: NODE_ROLE must be 'head' or 'worker'"
    exit 1
    ;;
esac
