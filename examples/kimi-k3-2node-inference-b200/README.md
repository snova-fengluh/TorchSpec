# Kimi-K3 2-Node Batch Inference (B200)

Inference-only 2-node setup for **Kimi-K3** (`moonshotai/Kimi-K3`): given the
prompts in a conversations JSONL dataset, generate a completion for each with
the target model. No training component, no draft model, no Mooncake — plain
vLLM batch generation spanning both nodes via Ray.

Default parallelism is **TP=8 x PP=2**: each node is one pipeline stage with
TP=8 inside it, so the per-layer TP all-reduces stay on intra-node NVLink and
only pipeline-stage activation handoffs cross the inter-node fabric. Set
`TP_SIZE=16 PP_SIZE=1` for cross-node TP instead (needs a fast inter-node
fabric to be worthwhile).

The shipped checkpoint is already MXFP4-packed (compressed-tensors,
`format=mxfp4-pack-quantized`); vLLM auto-detects it — do **not** pass an
explicit `quantization` arg.

## Prerequisites

- 2 nodes with 16 GPUs total (8x B200 each):
  - Node 0 (head): Ray head + vLLM driver + pipeline stage 0 (TP=8)
  - Node 1 (worker): pipeline stage 1 (TP=8)
- Model weights accessible on both nodes (~1.45 TB MXFP4 on disk,
  `trust_remote_code=true`). First load from cold NFS takes hours; page-cached
  reruns take minutes.
- **vLLM nightly** — PyPI 0.26.0 has no Kimi-K3 (nightly ships
  `vllm/models/kimi_k3/`). `ray_env.sh` defaults to the known-good shared
  conda env `vllm_kimi_k3`; override `CONDA_SH`/`CONDA_ENV` to use your own.
  Because the env lives on shared scratch, both nodes automatically run the
  identical python/torch/NCCL (required — mismatched NCCL versions hang
  silently in `ncclCommInitRank`).
- A control NIC that routes between the nodes (same /24 on both).
  `ray_env.sh` auto-detects it from `SUBNET_PREFIX` (default `10.1.33.`).
  NCCL runs over TCP on that NIC — no IB/RDMA/Mooncake needed.

See `examples/multi_node_ray_setup.md` for the full recipe & troubleshooting.

## How to run

### 1. Start the Ray cluster

On the **head node** (node 0):

```bash
NODE_ROLE=head bash examples/kimi-k3-2node-inference-b200/setup_ray_cluster.sh
```

It prints its control-NIC IP (`CTRL_IP`) and a preflight block (python/ray
paths, torch/NCCL/CUDA versions) — use the IP as `<node0_ip>` below.

On the **worker node** (node 1):

```bash
NODE_ROLE=worker HEAD_IP=<node0_ip> \
  bash examples/kimi-k3-2node-inference-b200/setup_ray_cluster.sh
```

Its preflight block must match the head's line-for-line.

Verify from the head node that all 16 GPUs joined:

```bash
ray status    # expect 2 nodes / 16 GPUs
```

### 2. Run inference

On the **head node**:

```bash
bash examples/kimi-k3-2node-inference-b200/run.sh
```

Defaults: model `/sms-scratch/vamsik/checkpoints/Kimi-K3`, dataset
`examples/data/sample_conversations.jsonl`, output
`outputs/kimi_k3_2node_inference/completions_<timestamp>.jsonl`.

## Data format

Input — one JSON object per line (same as `examples/data/*.jsonl`; `messages`
is accepted as an alias for `conversations`):

```json
{"id": "local_000000", "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Each conversation is cut at the **first assistant turn**; the prefix (system +
user turns) is formatted with the checkpoint's own chat template
(`add_generation_prompt=True`) and the model generates the completion. Any
existing assistant content in the file is ignored.

Output — one JSON object per line, input order preserved:

```json
{"id": "...", "prompt_messages": [...], "completion": "...", "finish_reason": "stop", "prompt_tokens": 123, "completion_tokens": 456}
```

Note: K3 assistant turns start inside the `<think>` channel, so completions
contain the thinking channel followed by the response channel. Special tokens
(channel markers like `<|close|>think<|sep|>`) are kept in the text by default
so the channels stay separable; pass `--skip-special-tokens` to strip them.

## Common customizations

```bash
# Custom dataset / output / model
DATA_PATH=path/to/prompts.jsonl OUTPUT_PATH=out.jsonl MODEL_PATH=/path/to/Kimi-K3 \
  bash examples/kimi-k3-2node-inference-b200/run.sh

# Greedy decoding, longer completions
TEMPERATURE=0 MAX_NEW_TOKENS=8192 bash examples/kimi-k3-2node-inference-b200/run.sh

# Extra flags pass through to generate.py
bash examples/kimi-k3-2node-inference-b200/run.sh --num-prompts 100 --max-num-seqs 32

# Cross-node TP instead of TP+PP (needs fast inter-node fabric)
TP_SIZE=16 PP_SIZE=1 bash examples/kimi-k3-2node-inference-b200/run.sh

# Degraded node (e.g. only 6-7 healthy GPUs): register fewer GPUs per node and
# shrink TP (TP must divide Kimi-K3's 96 attention heads: 16, 12, 8, 6, ...).
#   On BOTH nodes:  NUM_GPUS=6 NODE_ROLE=... bash .../setup_ray_cluster.sh
#   On the head:    TP_SIZE=6 PP_SIZE=2 bash .../run.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| `ray_env.sh` | Shared env (conda activate, NIC/NCCL config, vLLM knobs) — sourced by both scripts, on both nodes |
| `setup_ray_cluster.sh` | Preflight + clean stale state + start Ray head/worker |
| `run.sh` | Launch batch inference (run on head node after the cluster is ready) |
| `generate.py` | Standalone vLLM prompt→completion script (driven by `run.sh`) |

## Notes

- **RAY_ADDRESS is required.** `run.sh` exports `RAY_ADDRESS=auto` before
  launching. Without it, vLLM's EngineCore subprocess calls `ray.init()` with
  no address and starts a NEW single-node Ray instance — the logs show
  `Started a local Ray instance` followed by `The number of required GPUs
  exceeds the total number of available GPUs in the placement group`, even
  though `ray status` shows all 16 GPUs. `auto` discovery reads
  `/tmp/ray/session_latest`, which is why `setup_ray_cluster.sh` uses Ray's
  default temp dir. If `auto` still fails, set it explicitly:
  `RAY_ADDRESS=<node0_ip>:6379`.
- **`VLLM_USE_DEEP_GEMM=0` is required** (set in `ray_env.sh`): DeepGEMM's
  JIT crashes on Kimi-K3's MXFP4 MoE; this falls back to the Triton path.
- **`VLLM_PORT` must stay unset** (`ray_env.sh` unsets it): vLLM's TCPStore
  port scan otherwise starts near port ~100 and dies with EACCES without root.
- **Parallel layout.** `TP_SIZE * PP_SIZE` must equal the total GPUs
  registered with Ray, and TP must divide the 96 attention heads.
  `--max-model-len` defaults to 20000 (matches the training config); prompts
  longer than `max_model_len - max_new_tokens` tokens are skipped with a
  warning.
- **Need hidden states instead?** If the goal is materializing target-model
  outputs (aux hidden states + completions) for *offline training*, use
  `python -m torchspec.offline.generate --config ... --output ...` — that path
  needs the Mooncake/RDMA stack and a training-capable config, unlike this
  example.
