# Kimi-K3 MXFP4 3-Node Training (B200)

Production 3-node setup for training an Eagle3 draft model for **Kimi-K3**
(`moonshotai/Kimi-K3`) — a 2.8T-param (104B active) multimodal MoE with a
KDA/MLA hybrid text tower. The shipped checkpoint is already MXFP4-packed; vLLM
auto-detects the compressed-tensors format from the checkpoint.

## Prerequisites

- 3 nodes with 24 GPUs total (8x B200 each):
  - Node 0 (head): 8 GPUs (GPU 0-7) for FSDP draft training
  - Node 1-2 (workers): 8 GPUs each for vLLM inference (TP=16, `nnodes=2`)
- Model access to `moonshotai/Kimi-K3` (`trust_remote_code=true`)
- RDMA network for Mooncake hidden-state transfer (or fall back to TCP)

## Config

Uses [`configs/vllm_kimi_k3_3node.yaml`](../../configs/vllm_kimi_k3_3node.yaml)
with draft model config
[`configs/draft_models/kimi_k3_eagle3_mla.json`](../../configs/draft_models/kimi_k3_eagle3_mla.json).

The YAML already encodes the full topology (`vllm.nnodes=2`, `tp_size=16`,
`inference_num_gpus=16`, `training_num_gpus_per_node=8`) and the aux hidden-state
layers `[4, 48, 88]` (all full-attention/MLA layers). Ray auto-negotiates the
addresses left `null` (`vllm.dist_init_addr`, `mooncake.master_server_address`,
`mooncake.metadata_server`, and the training `MASTER_ADDR`).

## Before you run — fill these in

1. **Network interface.** The scripts default to `eth0`. Find yours with
   `ip -o -4 addr` and export it consistently on every node:
   ```bash
   export NCCL_SOCKET_IFNAME=<iface>   # e.g. bond0, ens0
   ```

2. **RDMA NIC names.** The YAML ships with placeholder `mooncake.device_name`
   (`mlx5_7,...`). Get the real device names with `ibdev2netdev -v` and pass them
   via `MOONCAKE_DEVICE_NAME` (see below), or edit the YAML. **No RDMA?** Set
   `MOONCAKE_PROTOCOL=tcp`.

3. **Model weights.** Ensure `moonshotai/Kimi-K3` is accessible (HF token or
   pre-downloaded, ~1.45 TB MXFP4 on disk).

## How to run

### 1. Start the Ray cluster

On the **head node** (node 0):

```bash
NCCL_SOCKET_IFNAME=<iface> NODE_ROLE=head \
  bash examples/kimi-k3-3node-b200/setup_ray_cluster.sh
```

It prints its `LOCAL_IP` — use that as `<node0_ip>` below.

On each **worker node** (nodes 1-2):

```bash
NCCL_SOCKET_IFNAME=<iface> HEAD_IP=<node0_ip> NODE_ROLE=worker \
  bash examples/kimi-k3-3node-b200/setup_ray_cluster.sh
```

Verify from the head node that all 24 GPUs joined:

```bash
ray status    # expect 3 nodes / 24 GPUs
```

### 2. Launch training

On the **head node**:

```bash
NCCL_SOCKET_IFNAME=<iface> MOONCAKE_DEVICE_NAME=<mlx5_...> \
  bash examples/kimi-k3-3node-b200/run.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| `setup_ray_cluster.sh` | Start Ray head/worker nodes |
| `run.sh` | Launch training (run on head node after the cluster is ready) |

## Common customizations

```bash
# No RDMA fabric — use TCP transport
MOONCAKE_PROTOCOL=tcp bash examples/kimi-k3-3node-b200/run.sh

# Override config file
CONFIG_FILE=path/to/custom.yaml bash examples/kimi-k3-3node-b200/run.sh

# Pass extra config overrides through to train_entry
bash examples/kimi-k3-3node-b200/run.sh training.learning_rate=1e-5
```

## Caveats — verify before a full run

- **Engine support (BLOCKER).** TorchSpec's `VllmEngine` +
  `MooncakeHiddenStatesConnector` path is unverified for the custom
  `KimiK3ForConditionalGeneration` (KDA/MLA hybrid) aux hidden-state extraction.
  Smoke-test that vLLM loads the model and extracts aux hidden states before
  committing to the full `num_epochs=3` run.
- **Memory footprint.** The 2-node (TP=16) inference layout assumes the native
  MXFP4 target fits with KV/activation headroom at `mem_fraction_static=0.85`.
  Re-verify the on-GPU size for your build before assuming 2 nodes suffice.
