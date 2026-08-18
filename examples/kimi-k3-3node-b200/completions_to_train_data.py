"""Convert 2-node inference completions into 3-node training conversations.

The training pipeline (torchspec.train_entry) does NOT generate text: the
target model runs a prefill-only forward pass over each conversation and the
draft model is trained on the assistant spans (loss mask). So the training
JSONL must already contain assistant completions — ideally Kimi-K3's own
(on-policy), which is exactly what
examples/kimi-k3-2node-inference-b200/generate.py produces.

This script turns that generator's output back into the ``conversations``
format the "kimi-k3" chat template expects:

Input (one JSON object per line, from generate.py):
    {"id": ..., "prompt_messages": [...], "completion": "...", ...}

Output (one JSON object per line):
    {"id": ..., "conversations": [
        ...prompt messages...,
        {"role": "assistant", "reasoning_content": "...", "content": "..."}]}

K3 completions start inside the <think> channel, so with the default
(--skip-special-tokens NOT passed) the text contains the raw XTML channel
markers: {think}<|close|>think<|sep|><|open|>response<|sep|>{response}...
We split on those markers into reasoning_content / content — the KimiK3Parser
re-renders both channels when formatting training sequences. If the markers
are absent (generation ran with --skip-special-tokens), the whole text becomes
``content`` and the reasoning channel is left empty.

Usage:
    python examples/kimi-k3-3node-b200/completions_to_train_data.py \
        --input outputs/kimi_k3_2node_inference/completions_<ts>.jsonl \
        --output examples/data/kimi_k3_train_conversations.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# <|close|>think<|sep|><|open|>response<|sep|> separates the two channels.
_CHANNEL_SPLIT = re.compile(r"<\|close\|>think<\|sep\|>\s*<\|open\|>response<\|sep\|>")
# Trailing structural markers after the response body.
_TRAILER = re.compile(
    r"(<\|close\|>response<\|sep\|>|<\|close\|>message<\|sep\|>|<\|end_of_msg\|>)+\s*$"
)


def split_completion(text: str) -> tuple[str, str]:
    """Return (reasoning, response) from a raw K3 completion."""
    parts = _CHANNEL_SPLIT.split(text, maxsplit=1)
    if len(parts) == 2:
        reasoning, response = parts
    else:
        reasoning, response = "", text
    response = _TRAILER.sub("", response)
    return reasoning.strip(), response.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="completions JSONL from generate.py")
    parser.add_argument("--output", required=True, help="training conversations JSONL")
    parser.add_argument(
        "--keep-unfinished",
        action="store_true",
        help="Keep completions with finish_reason != 'stop' (truncated by max tokens). "
        "Dropped by default: truncated tails teach the draft mid-sentence endings.",
    )
    args = parser.parse_args()

    written = skipped = 0
    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin):
            if line_no >= 1031:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            completion = obj.get("completion", "")
            if not completion:
                skipped += 1
                continue
            if not args.keep_unfinished and obj.get("finish_reason") not in (None, "stop"):
                skipped += 1
                continue

            reasoning, response = split_completion(completion)
            if not reasoning and not response:
                skipped += 1
                continue

            assistant: dict = {"role": "assistant", "content": response}
            if reasoning:
                assistant["reasoning_content"] = reasoning

            fout.write(
                json.dumps(
                    {
                        "id": obj.get("id", f"line_{line_no}"),
                        "conversations": [*obj["prompt_messages"], assistant],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"Wrote {written} conversations to {args.output} ({skipped} skipped)", file=sys.stderr)


if __name__ == "__main__":
    main()
