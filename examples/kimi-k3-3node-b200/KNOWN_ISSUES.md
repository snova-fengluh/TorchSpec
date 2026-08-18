# Kimi-K3 3-Node Training — Known Issues & Road to a Smooth Run

Debugging record for bringing up `examples/kimi-k3-3node-b200` (2 inference
nodes with cross-node vLLM TP=16 + 1 FSDP training node). Last updated
**2026-08-17**.

## Why this path is hard (and the 2-node inference example isn't)

The working [`kimi-k3-2node-inference-b200`](../kimi-k3-2node-inference-b200)
example runs **TP=8 × PP=2**: every tensor-parallel all-reduce stays on
intra-node NVLink; only small pipeline activations cross the TCP link, via
vLLM's mature `ray` executor.

This training path forces **TP=16 across two nodes** (TorchSpec's multi-node
vLLM computes `tp_size = nnodes × gpus_per_node`; there is no PP option), over
vLLM's newer multi-node `mp` executor. That drags every per-layer collective
onto the slow TCP fabric and pushes every single-node-only fast path (CUDA
IPC, NVSwitch multicast, symmetric memory) outside its supported regime.
Nearly every issue below is a downstream consequence.

## Issue log (chronological, all root-caused)

### 1. Mooncake master exits 127 — `libcudart.so.12` not found  — FIXED
`mooncake_master` links against the CUDA-12 runtime; the venv's
`nvidia.cuda_runtime` is a **namespace package** (`__file__ = None`), so the
old auto-detection one-liner silently failed and the dir never reached
`LD_LIBRARY_PATH`. Trainer then died with `ConnectionRefusedError` (downstream
symptom — it was connecting to the master that never started).
**Fix:** `ray_env.sh` resolves the path via `importlib.util.find_spec`
search locations. Must be exported **before `ray start`** (raylet snapshots
its env; `LD_LIBRARY_PATH` is not forwardable via `runtime_env`).

### 2. `/dev/shm` model path breaks the driver/trainer — FIXED
`target_model_path: /dev/shm/Kimi-K3` made the driver (tokenizer,
draft-config generation) and trainer (embed/lm_head/norm) read a **node-local**
path that only exists on inference nodes → transformers fell back to treating
it as a HF repo id (`Repo id must be in the form ...`).
**Fix:** new `model.inference_model_path` config field
(`torchspec/config/train_config.py`) + a central swap in
`torchspec/inference/factory.py:_resolve_engine_args` — engines load from the
staged `/dev/shm` copy, driver/trainer keep the shared-FS
`target_model_path`. `/dev/shm/Kimi-K3` (~1.45 TB, counts against RAM) must be
staged on **every inference node**.

### 3. Gloo crash `ss1.ss_family == ss2.ss_family. 10 vs 2` — FIXED
`get_torchspec_env_vars()` forwards the **driver's**
`NCCL/GLOO/TP_SOCKET_IFNAME` into every actor's `runtime_env`, overriding the
correct per-node raylet values. NIC names differ across nodes; on the other
node the forwarded name matched an **IPv6-only interface** (down RoCE rail
with only `fe80::`), so gloo advertised IPv6 while peers advertised IPv4 —
family mismatch at connect (`Device::isInitiator`).
**Fix:** `torchspec/utils/env.py:resolve_local_socket_ifnames()`, called at
the top of `VllmEngine.init()` and `TrainerActor.init()` — rewrites the vars
to the interface owning the node's Ray IP unless the current value is a
deliberate config (comma list, `^`/`=` syntax, or a valid local IPv4 iface).

### 4. `Failed to send fd: No such file or directory` (symm_mem) — PATCHED IN VENV ⚠
Kimi-K3's latent-MoE tail fusion (CuTe-DSL fused allreduce+RMSNorm)
auto-enables on SM100 for TP∈{8,16} and builds its workspace with **torch
symmetric memory** — CUDA-IPC fd exchange over local unix sockets, intra-node
only. With TP=16 spanning nodes, the fd send targets a socket on the other
machine.
**Fix (vLLM patch):** `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` —
disable `enable_k3_latent_moe_tail_fusion` when `parallel_config.nnodes > 1`
(falls back to the native latent-MoE path, correct at any TP size).
⚠ Lives in the venv's site-packages (`.orig` backup alongside); **a vLLM
upgrade silently reverts it**. Worth upstreaming.

### 5. `collective_rpc should not be called on follower node` — FIXED
In vLLM's multi-node `mp` design only the **leader** node (node_rank 0) runs a
real engine; follower nodes must run *only* a `MultiprocExecutor` whose
workers join the leader's broadcast queue (what `vllm serve --headless` does).
TorchSpec's `VllmEngine` called `LLM()` on every node, so the follower built a
full EngineCore and hit the leader-only KV-cache init.
**Fix:** `torchspec/inference/engine/vllm_engine.py:_init_engine` — on
`nnodes > 1 and node_rank > 0`, build the identical `VllmConfig` via
`EngineArgs(**engine_kwargs).create_engine_config(headless=True)` and start
`MultiprocExecutor(vllm_config, monitor_workers=False)` + a background worker
monitor instead of `LLM()`.

### 6. FlashInfer all-reduce `mnnvl` bootstrap deadlock — PATCHED IN VENV ⚠
On multi-node, vLLM auto-selects flashinfer's **mnnvl** all-reduce backend,
which needs a cross-node NVLink/NVSwitch multicast fabric (GB200-style). On
plain Ethernet its collective bootstrap partially times out (`Application
timeout caused pair closure`): some ranks fall back to NCCL while others stay
blocked in the bootstrap **forever** (not covered by any watchdog — vLLM TP
runs on its own pynccl comms). K3 calls this per layer via
`fused_allreduce_rms_norm`.
**Fix (vLLM patch):**
`vllm/distributed/device_communicators/flashinfer_all_reduce.py` — both
workspace getters return `None` on `get_node_count() > 1` unless
`VLLM_FLASHINFER_ALLREDUCE_BACKEND` is set explicitly → clean NCCL fallback
everywhere. ⚠ Same caveat: reverted by a vLLM reinstall; upstreamable.

### 7. Rank frozen in Triton `_init_handles` (lazy module load vs NCCL) — MITIGATED, PENDING VERIFICATION
py-spy showed the leader's rank hard-blocked in `cuModuleLoadData` (Triton
kernel first-launch) while its GPUs spun in pending NCCL all-reduces and the
follower — 93 layers ahead thanks to a warm kernel cache — waited at the
sampler all-gather. CUDA **lazy module loading** (default since 12.2) can
deadlock a module load against spinning NCCL kernels on the same device; the
multi-second TCP collectives plus cache-skewed ranks make the collision
window enormous.
**Mitigation:** `ray_env.sh` now sets `CUDA_MODULE_LOADING=EAGER` (documented
workaround) and `NCCL_SOCKET_NTHREADS=4` / `NCCL_NSOCKS_PERTHREAD=2` (more TCP
streams → shorter collectives → smaller windows). Needs a cluster restart to
take effect. **Fallback if it recurs:** level the cache skew —
`mv ~/.triton/cache ~/.triton/cache.bak` (shared home) so both nodes JIT in
lockstep.

## Current status / what is still unverified

The run has never yet completed vLLM's **memory-profiling forward pass** (the
first full model execution). Milestones to watch in order:

1. `Available KV cache memory: ... GiB` — profile forward done (issue 7 fix
   pending verification here).
2. Engine init completes; TorchSpec logs `Successfully initialized ... Vllm
   engines`.
3. **BLOCKER 5 (from the YAML):** aux hidden-state extraction
   (`extract_hidden_states` + `MooncakeHiddenStatesConnector`) on the KDA/MLA
   hybrid — completely unverified; failure here would show as missing
   `kv_transfer_params` / empty mooncake results.
4. Mooncake hidden-state transfer over TCP to the training node — unverified.
5. First training step (loss mask, TTT) — unverified. Smoke-test with
   `training.num_train_steps=3` before any long run.

## Structural fixes for an eventually smooth run (ranked)

1. **Rail-pinned RoCE between the two inference nodes** (env-only, biggest
   win). Cross-node TP over TCP (~1.8 Gb/s measured) makes every prefill
   minutes-slow and widens every race window; issues 6–7 are far less likely
   at ~60 GB/s. The system-order-LCS `NCCL_IB_HCA` procedure was validated on
   other node pairs; needs per-pair rail mapping and the raylet memlock fix
   (`sudo prlimit --memlock=unlimited:unlimited --pid $(pgrep -x raylet)`
   after **every** `ray start` — it does not survive restarts).
2. **Upstream the two vLLM patches** (issues 4 and 6) so venv upgrades don't
   silently reintroduce the deadlocks: both are "gate NVLink-fabric fast paths
   on `nnodes == 1`" one-liners.
3. **PP support in TorchSpec's vLLM engine** — would reproduce the 2-node
   example's TP-on-NVLink layout, eliminating this problem class. Currently
   blocked upstream: vLLM's spec-decode/aux-hidden-state machinery only runs
   on the **last** PP stage (`gpu_model_runner` gates on
   `get_pp_group().is_last_rank`), while K3's aux layers [4, 48, 88] span
   stages; the Mooncake connector would also need per-stage assembly.
4. **Keep the JIT caches warm** — after one successful profile pass, both
   nodes' Triton caches (shared NFS home) make subsequent bring-ups fast.
   Avoid SIGKILLing mid-JIT when possible.

## Files changed (this effort)

| File | Change |
|------|--------|
| `examples/kimi-k3-3node-b200/ray_env.sh` | cudart detection, per-node NIC autodetect, NCCL-over-TCP, `CUDA_MODULE_LOADING=EAGER`, NCCL socket streams |
| `torchspec/utils/env.py` | `resolve_local_socket_ifnames()` |
| `torchspec/inference/engine/vllm_engine.py` | per-node iface fix call; headless follower executor |
| `torchspec/training/trainer_actor.py` | per-node iface fix call |
| `torchspec/config/train_config.py` | `model.inference_model_path` |
| `torchspec/inference/factory.py` | `_resolve_engine_args` path swap |
| `configs/vllm_kimi_k3_3node.yaml` | split model paths, `limit_mm_per_prompt: {image: 0}` |
| `venv/.../vllm/models/kimi_k3/nvidia/latent_moe_runner.py` | ⚠ nnodes guard (patch, `.orig` backup) |
| `venv/.../vllm/distributed/device_communicators/flashinfer_all_reduce.py` | ⚠ multi-node guard (patch, `.orig` backup) |
