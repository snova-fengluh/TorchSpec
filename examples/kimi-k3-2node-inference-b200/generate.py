"""Batch inference for Kimi-K3 on a 2-node Ray cluster with vLLM.

Given a conversations JSONL dataset (same format as examples/data/*.jsonl),
cut each conversation at the first assistant turn, apply the model's own chat
template, and generate a completion for every prompt with the target model.
No training component — plain prompt -> completion generation.

Runs on the Ray head node; vLLM spans both nodes via
distributed_executor_backend="ray" (TP = --tp across all registered GPUs).

Input (one JSON object per line):
    {"id": "...", "conversations": [{"role": "user", "content": "..."}, ...]}
    ("messages" is accepted as an alias for "conversations".)

Output (one JSON object per line, order preserved):
    {"id": ..., "prompt_messages": [...], "completion": "...",
     "finish_reason": "...", "prompt_tokens": N, "completion_tokens": N}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model path (Kimi-K3 checkpoint)")
    parser.add_argument("--input", required=True, help="Conversations JSONL dataset")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size (must divide Kimi-K3's 96 attention heads: 16, 12, 8, ...). Keep <= GPUs per node so TP all-reduces stay on NVLink.")
    parser.add_argument("--pp", type=int, default=2, help="Pipeline parallel size (number of pipeline stages, typically = number of nodes). tp * pp must equal the total GPUs in the Ray cluster.")
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-prompts", type=int, default=None, help="Only generate for the first N prompts (default: all)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Max concurrent sequences in a vLLM batch")
    parser.add_argument("--chunk-size", type=int, default=100, help="Generate and flush results this many prompts at a time, so partial output survives a crash. Keep >= max-num-seqs to avoid starving the vLLM batch.")
    parser.add_argument("--skip-special-tokens", action="store_true", help="Strip special tokens from completions. Off by default so the <think>/<response> channel markers survive in the output text.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile/CUDA graphs (the verified multi-node config). Omit for peak throughput once the setup is stable.")
    return parser.parse_args()


def load_prompt_messages(path: str, limit: int | None):
    """Yield (id, prompt_messages) — the conversation prefix before the first
    assistant turn, i.e. what the target model should complete."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get("conversations") or obj.get("messages")
            if not messages:
                print(f"WARNING: line {line_no}: no conversations/messages key, skipped", file=sys.stderr)
                continue
            prefix = []
            for msg in messages:
                # if msg["role"] == "assistant":
                #     break
                # prefix.append({"role": msg["role"], "content": msg["content"]})
                if msg["from"] == "gpt":
                    break
                elif msg["from"] == "system":
                    prefix.append({"role": "system", "content": msg["value"]})
                elif msg["from"] == "human":
                    prefix.append({"role": "user", "content": msg["value"]})
                else:
                    breakpoint()
            if not prefix:
                print(f"WARNING: line {line_no}: conversation starts with an assistant turn, skipped", file=sys.stderr)
                continue
            records.append((obj.get("id", f"line_{line_no}"), prefix))
            if limit is not None and len(records) >= limit:
                break
    return records


def connect_to_ray_cluster(required_gpus: int) -> None:
    """Join the existing multi-node Ray cluster and fail fast if it is too small.

    vLLM's EngineCore runs in a subprocess; without RAY_ADDRESS in the
    environment its ray.init() starts a brand-new LOCAL Ray instance that only
    sees this node's GPUs ("Started a local Ray instance" in the logs, followed
    by placement-group allocation failures). Exporting RAY_ADDRESS makes both
    this driver and the EngineCore subprocess connect to the real cluster.
    """
    import ray

    ray_address = os.environ.get("RAY_ADDRESS", "auto")
    os.environ["RAY_ADDRESS"] = ray_address  # inherited by the EngineCore subprocess
    ray.init(address=ray_address)
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    if total_gpus < required_gpus:
        raise SystemExit(
            f"Ray cluster has {total_gpus} GPUs but tp*pp={required_gpus} are required. "
            "Did the worker node join? Check `ray status`."
        )
    print(f"Connected to Ray cluster at {ray_address}: {total_gpus} GPUs available")


def main():
    args = parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    records = load_prompt_messages(args.input, args.num_prompts)
    if not records:
        raise SystemExit(f"No usable prompts found in {args.input}")
    print(f"Loaded {len(records)} prompts from {args.input}")

    # Tokenize with the checkpoint's own tokenizer + chat template (K3 ships a
    # custom tiktoken-based XTML format via tokenization_kimi.py). Passing token
    # ids to vLLM avoids any re-tokenization of special-token strings.
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_budget = args.max_model_len - args.max_new_tokens
    prompts, kept = [], []
    for record_id, messages in records:
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(token_ids) > prompt_budget:
            print(f"WARNING: {record_id}: prompt has {len(token_ids)} tokens > budget {prompt_budget}, skipped", file=sys.stderr)
            continue
        prompts.append({"prompt_token_ids": token_ids})
        kept.append((record_id, messages))
    print(f"Tokenized {len(kept)} prompts ({len(records) - len(kept)} skipped)")

    connect_to_ray_cluster(required_gpus=args.tp * args.pp)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        distributed_executor_backend="ray",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        # Text-only run: skip multimodal memory profiling so it doesn't eat
        # into the KV cache budget (matches the verified K3 config).
        limit_mm_per_prompt={"image": 0},
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=args.skip_special_tokens,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate in chunks and flush each one, so a crash / preemption late in a
    # long run doesn't throw away everything generated so far.
    chunk_size = max(1, args.chunk_size)
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for start in range(0, len(prompts), chunk_size):
            chunk_prompts = prompts[start : start + chunk_size]
            chunk_kept = kept[start : start + chunk_size]
            print(
                f"Generating prompts {start}-{start + len(chunk_prompts) - 1} "
                f"of {len(prompts)}",
                flush=True,
            )
            outputs = llm.generate(chunk_prompts, sampling_params)
            for (record_id, messages), out in zip(chunk_kept, outputs):
                completion = out.outputs[0]
                f.write(
                    json.dumps(
                        {
                            "id": record_id,
                            "prompt_messages": messages,
                            "completion": completion.text,
                            "finish_reason": completion.finish_reason,
                            "prompt_tokens": len(out.prompt_token_ids),
                            "completion_tokens": len(completion.token_ids),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.flush()
            os.fsync(f.fileno())
            written += len(chunk_kept)
            print(f"Wrote {written}/{len(prompts)} completions to {output_path}", flush=True)

    print(f"Wrote {written} completions to {output_path}")


if __name__ == "__main__":
    main()
