#!/bin/bash
# Ray cluster setup for Kimi-K3 MXFP4 Eagle3 3-node training (B200)
#
# Node layout (3 nodes x 8 B200 = 24 GPUs):
#   Node 0 (head):   Ray head + training, 8 GPUs (GPU 0-7)
#   Node 1 (worker): vLLM inference, 8 GPUs (TP=16, node_rank=0)
#   Node 2 (worker): vLLM inference, 8 GPUs (TP=16, node_rank=1)
#
# Usage — run on each node with the appropriate NODE_ROLE:
#
#   # Node 0 (Ray head + training):
#   NODE_ROLE=head bash examples/kimi-k3-3node-b200/setup_ray_cluster.sh
#
#   # Node 1 & 2 (inference workers):
#   HEAD_IP=<node0_ip> NODE_ROLE=worker bash examples/kimi-k3-3node-b200/setup_ray_cluster.sh
#
# Environment variables:
#   HEAD_IP             — IP of node 0 (Ray head). Required only for worker nodes.
#   NODE_ROLE           — "head" | "worker"
#   RAY_PORT            — Ray GCS port (default: 6380)
#   NCCL_SOCKET_IFNAME  — network interface for NCCL/Gloo/TP (default: eth0).
#                         Find yours with: ip -o -4 addr
#   MOONCAKE_CUDART_DIR — dir containing libcudart.so.12 for the mooncake_master
#                         binary. Auto-detected from the venv's pip CUDA runtime
#                         (nvidia.cuda_runtime) if unset. See Error 6 in SETUP_GUIDE.md.

set -euo pipefail
set -x

# Network interface used by NCCL/Gloo/TP collectives. Override to match your
# fabric (e.g. bond0, ens0). Find it with: ip -o -4 addr
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# The mooncake_master binary links against libcudart.so.12 (CUDA 12 ABI). On
# nodes whose system CUDA is 13.x (only libcudart.so.13 present) it exits 127
# with "error while loading shared libraries: libcudart.so.12" — see Error 6 in
# SETUP_GUIDE.md. The CUDA-12 runtime ships inside the venv's pip package. It
# must be on LD_LIBRARY_PATH BEFORE `ray start` so the raylet — and the
# mooncake_master Ray actor it spawns — inherit it (LD_LIBRARY_PATH is NOT
# forwarded via runtime_env, so exporting it only in run.sh does not work).
MOONCAKE_CUDART_DIR="${MOONCAKE_CUDART_DIR:-$(python3 -c "import os, nvidia.cuda_runtime as m; print(os.path.join(os.path.dirname(m.__file__), 'lib'))" 2>/dev/null || true)}"
if [[ -n "$MOONCAKE_CUDART_DIR" && -e "$MOONCAKE_CUDART_DIR/libcudart.so.12" ]]; then
  export LD_LIBRARY_PATH="$MOONCAKE_CUDART_DIR:${LD_LIBRARY_PATH:-}"
  echo "Prepended libcudart.so.12 dir to LD_LIBRARY_PATH: $MOONCAKE_CUDART_DIR"
else
  echo "WARNING: libcudart.so.12 not found (looked in '${MOONCAKE_CUDART_DIR:-<empty>}')."
  echo "         mooncake_master may fail with exit code 127 — set MOONCAKE_CUDART_DIR"
  echo "         to the dir containing libcudart.so.12. See Error 6 in SETUP_GUIDE.md."
fi

NODE_ROLE="${NODE_ROLE:?NODE_ROLE must be set to head or worker}"
RAY_PORT="${RAY_PORT:-6380}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_torchspec_kimi_k3_$(id -u)}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/kimi_k3_3node_ray_${NODE_ROLE}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")

echo "=============================================="
echo "Kimi-K3 MXFP4 3-Node Ray Cluster (B200)"
echo "NODE_ROLE: $NODE_ROLE  LOCAL_IP: $LOCAL_IP"
echo "IFNAME: $NCCL_SOCKET_IFNAME"
echo "=============================================="

case "$NODE_ROLE" in
  head)
    echo "=== Starting Ray HEAD node (training, 8 GPUs, GPU 0-7) ==="
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    ray stop --force 2>/dev/null || true
    ray start --head \
      --port "$RAY_PORT" \
      --temp-dir "$RAY_TEMP_DIR" \
      --num-gpus 8 \
      --disable-usage-stats
    echo "Ray head started at $LOCAL_IP:$RAY_PORT"
    echo "Next steps:"
    echo "  Worker nodes: HEAD_IP=$LOCAL_IP NODE_ROLE=worker bash examples/kimi-k3-3node-b200/setup_ray_cluster.sh"
    echo "  Training:     bash examples/kimi-k3-3node-b200/run.sh"
    ;;

  worker)
    HEAD_IP="${HEAD_IP:?HEAD_IP must be set to node 0 IP address}"
    RAY_ADDR="${HEAD_IP}:${RAY_PORT}"
    export RAY_ADDRESS="$RAY_ADDR"
    echo "=== Joining Ray cluster as WORKER (inference, 8 GPUs) ==="
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    ray stop --force 2>/dev/null || true
    ray start \
      --address "$RAY_ADDR" \
      --temp-dir "$RAY_TEMP_DIR" \
      --num-gpus 8 \
      --disable-usage-stats
    echo "Worker joined Ray cluster at $RAY_ADDR"
    ;;

  *)
    echo "ERROR: NODE_ROLE must be 'head' or 'worker'"
    echo "  head   — Ray head + training node (8 GPUs, GPU 0-7)"
    echo "  worker — inference worker node (8 GPUs)"
    exit 1
    ;;
esac
