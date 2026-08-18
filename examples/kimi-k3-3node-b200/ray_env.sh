#!/bin/bash
# Shared environment for Kimi-K3 3-node training (2 inference + 1 training).
# Source this on ALL THREE nodes BEFORE `ray start` — the raylet inherits the
# environment and propagates it to every actor it spawns (vLLM workers,
# training actors, mooncake_master). Adapted from the verified
# examples/kimi-k3-2node-inference-b200/ray_env.sh, plus the Mooncake bits
# training needs (libcudart.so.12 on LD_LIBRARY_PATH, memlock).
#
#   source examples/kimi-k3-3node-b200/ray_env.sh
#
# Override points (export before sourcing):
#   VENV_ACTIVATE — python env activate script. Default is the repo's ./venv,
#       which has torchspec (editable) + mooncake-transfer-engine + vLLM
#       NIGHTLY with Kimi-K3 support. The shared `vllm_kimi_k3` conda env used
#       by the 2-node inference example is NOT enough here — it lacks
#       omegaconf/mooncake, so train_entry would crash. Because the venv lives
#       on shared scratch, all nodes automatically get the identical
#       python/torch/NCCL (mismatched NCCL versions hang silently in
#       ncclCommInitRank).
#   NET_IF_CTRL   — cross-node control NIC. Auto-detected as the interface
#       holding this node's ${SUBNET_PREFIX}x address; override if that fails.
#   SUBNET_PREFIX — cross-node /24 the nodes share (default 10.1.33.).
#   MOONCAKE_CUDART_DIR — dir with libcudart.so.12 for the mooncake_master
#       binary. Auto-detected from the venv's pip CUDA runtime if unset.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ---- interpreter: repo venv (torchspec + mooncake + vLLM nightly, on shared scratch) ----
# VENV_ACTIVATE="${VENV_ACTIVATE:-$ROOT_DIR/venv/bin/activate}"
# if [[ ! -f "$VENV_ACTIVATE" ]]; then
#   echo "ERROR: venv activate script not found: $VENV_ACTIVATE" >&2
#   echo "Set VENV_ACTIVATE to a python env with torchspec, mooncake and vLLM nightly." >&2
#   return 1 2>/dev/null || exit 1
# fi
# source "$VENV_ACTIVATE"

VENV_ACTIVATE="${VENV_ACTIVATE:-$ROOT_DIR/venv}"
source ~/miniconda3/bin/activate
conda activate "$VENV_ACTIVATE"

hash -r   # flush cached command lookups so python/ray resolve into this env

# ---- HF cache on scratch ----
export HF_HOME="${HF_HOME:-/sms-scratch/${USER}/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# ---- cross-node control NIC ----
# All three nodes share a /24 (10.1.33.20 / .22 / .24); the control NIC is
# whichever interface owns this node's address there.
SUBNET_PREFIX="${SUBNET_PREFIX:-10.1.33.}"
if [[ -z "${NET_IF_CTRL:-}" ]]; then
  NET_IF_CTRL=$(ip -o -4 addr | awk -v p="$SUBNET_PREFIX" 'index($4, p) == 1 {print $2; exit}')
fi
if [[ -z "${NET_IF_CTRL:-}" ]]; then
  echo "ERROR: no interface with a ${SUBNET_PREFIX}* address found." >&2
  echo "List NICs with: ip -o -4 addr   then: export NET_IF_CTRL=<iface>" >&2
  return 1 2>/dev/null || exit 1
fi
export NET_IF_CTRL
LOCAL_CTRL_IP=$(ip -o -4 addr show "$NET_IF_CTRL" | awk '{print $4}' | cut -d/ -f1)
export LOCAL_CTRL_IP

# NCCL over TCP on the control NIC — the verified 2-node inference config.
# This cluster has no shared IB/RoCE fabric usable across arbitrary node
# pairs, so force the plain-socket path; NVLink stays on for intra-node
# TP/FSDP. (Rail-pinned RoCE only worked on specific node pairs and caused
# ibv_modify_qp timeouts elsewhere — see examples/multi_node_ray_setup.md.)
export NCCL_SOCKET_IFNAME="$NET_IF_CTRL"
export GLOO_SOCKET_IFNAME="$NET_IF_CTRL"
export TP_SOCKET_IFNAME="$NET_IF_CTRL"
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_IBEXT_DISABLE=1
export NCCL_NET_PLUGIN=none
export NCCL_P2P_DISABLE=0
# NCCL's socket transport defaults to very few TCP streams; more streams
# substantially raise cross-node allreduce throughput on a fat control NIC.
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-2}"

# CUDA lazy module loading (default since CUDA 12.2) can deadlock a
# cuModuleLoad (e.g. Triton kernel first-launch _init_handles) against
# pending/spinning NCCL kernels on the same device — observed as one rank
# frozen in triton _init_handles while all peers wait in a collective.
# Eager loading moves module loads to init time, before collectives fly.
export CUDA_MODULE_LOADING=EAGER

# ---- vLLM knobs (verified in the 2-node inference example) ----
# DeepGEMM JIT crashes on Kimi-K3's MXFP4 MoE — force the Triton fallback.
export VLLM_USE_DEEP_GEMM=0
# vLLM's TCPStore port scan defaults to a base near ~100, which needs root.
# Unset → vLLM uses bind(("", 0)) and the OS picks a free ephemeral port.
unset VLLM_PORT
export VLLM_HAS_FLASHINFER_CUBIN=1
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-/sms-scratch/${USER}/flashinfer_cache}"
mkdir -p "$FLASHINFER_CACHE_DIR"

# ---- Mooncake (hidden-state transfer, training only) ----
# The mooncake_master binary links against libcudart.so.12 (CUDA 12 ABI). On
# nodes whose system CUDA is 13.x it exits 127 with "error while loading
# shared libraries: libcudart.so.12". The CUDA-12 runtime ships inside the
# venv's pip package; it must be on LD_LIBRARY_PATH BEFORE `ray start` so the
# raylet — and the mooncake_master Ray actor it spawns — inherit it
# (LD_LIBRARY_PATH is NOT forwarded via runtime_env).
if [[ -z "${MOONCAKE_CUDART_DIR:-}" ]]; then
  # NB: nvidia.cuda_runtime is a NAMESPACE package (no __file__) in this venv,
  # so resolve it via find_spec's search locations instead of module.__file__.
  MOONCAKE_CUDART_DIR=$(python3 - 2>/dev/null <<'PY' || true
import importlib.util, os
spec = importlib.util.find_spec("nvidia.cuda_runtime")
for loc in (spec.submodule_search_locations or []) if spec else []:
    lib = os.path.join(loc, "lib")
    if os.path.exists(os.path.join(lib, "libcudart.so.12")):
        print(lib)
        break
PY
)
fi
if [[ -n "${MOONCAKE_CUDART_DIR:-}" && -e "$MOONCAKE_CUDART_DIR/libcudart.so.12" ]]; then
  export LD_LIBRARY_PATH="$MOONCAKE_CUDART_DIR:${LD_LIBRARY_PATH:-}"
else
  echo "WARNING: libcudart.so.12 not found (looked in '${MOONCAKE_CUDART_DIR:-<empty>}')." >&2
  echo "         mooncake_master may exit 127 — set MOONCAKE_CUDART_DIR." >&2
fi
# Mooncake RDMA needs unlimited locked memory on the raylet; harmless for TCP.
ulimit -l unlimited 2>/dev/null || true

echo "env ready: python=$(which python)"
echo "           NET_IF_CTRL=${NET_IF_CTRL}  LOCAL_CTRL_IP=${LOCAL_CTRL_IP}"
