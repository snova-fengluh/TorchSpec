#!/bin/bash
# Shared environment for Kimi-K3 2-node inference. Source this on BOTH nodes
# BEFORE `ray start` — the raylet inherits the environment and propagates it
# to every vLLM worker it spawns (see examples/multi_node_ray_setup.md).
#
#   source examples/kimi-k3-2node-inference-b200/ray_env.sh
#
# Override points (export before sourcing):
#   CONDA_SH / CONDA_ENV — python env that can import vLLM. Kimi-K3 needs
#       vLLM NIGHTLY (PyPI 0.26.0 has no kimi_k3 model); the default env
#       below lives on shared scratch, so both nodes automatically get the
#       identical python/torch/NCCL — the env-parity preflight passes for free.
#   NET_IF_CTRL   — cross-node control NIC. Auto-detected as the interface
#       holding this node's ${SUBNET_PREFIX}x address; override if that fails.
#   SUBNET_PREFIX — cross-node /24 the two nodes share (default 10.1.33.).

# ---- interpreter: vLLM-nightly env (known-good, on shared scratch) ----
CONDA_SH="${CONDA_SH:-/sms-scratch/badreddinen/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-vllm_kimi_k3}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
hash -r   # flush cached command lookups so python/ray resolve into this env

# ---- HF cache on scratch ----
export HF_HOME="${HF_HOME:-/sms-scratch/${USER}/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# ---- cross-node control NIC ----
# The two nodes share a /24 (e.g. 10.1.33.26 + 10.1.33.12 on 10.1.33.0/24);
# the control NIC is whichever interface owns this node's address there.
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

# NCCL over TCP on the control NIC. This cluster has no shared IB/RoCE fabric
# usable cross-node, so force the plain-socket path; NVLink stays on for
# intra-node TP. (Rail-pinned RoCE only worked on specific node pairs and is
# exactly what caused the earlier ibv_modify_qp timeouts — don't re-enable
# without verifying a shared fabric per multi_node_ray_setup.md N.B.2.)
export NCCL_SOCKET_IFNAME="$NET_IF_CTRL"
export GLOO_SOCKET_IFNAME="$NET_IF_CTRL"
export TP_SOCKET_IFNAME="$NET_IF_CTRL"
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_IBEXT_DISABLE=1
export NCCL_NET_PLUGIN=none
export NCCL_P2P_DISABLE=0

# ---- vLLM knobs ----
# DeepGEMM JIT crashes on Kimi-K3's MXFP4 MoE — force the Triton fallback.
export ABSMAX_OUT="${ABSMAX_OUT:-/sms-scratch/${USER}/absmax_dumps}"
export VLLM_USE_DEEP_GEMM=0
# vLLM's TCPStore port scan defaults to a base near ~100, which needs root.
# Unset → vLLM uses bind(("", 0)) and the OS picks a free ephemeral port.
unset VLLM_PORT
export VLLM_HAS_FLASHINFER_CUBIN=1
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-/sms-scratch/${USER}/flashinfer_cache}"
mkdir -p "$FLASHINFER_CACHE_DIR"

echo "env ready: python=$(which python)"
echo "           NET_IF_CTRL=${NET_IF_CTRL}  LOCAL_CTRL_IP=${LOCAL_CTRL_IP}"
