#!/usr/bin/env python3
"""Verify one Qwen3.5 mode through the real OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from openai import OpenAI


ANSWER_RE = re.compile(r"(?i)\bANSWER\s*:\s*\(?([A-J])\)?")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("base", "thinking", "non_thinking"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.temperature is None:
        args.temperature = 1.0 if args.mode == "thinking" else 0.7
    if args.top_p is None:
        args.top_p = 0.95 if args.mode == "thinking" else 0.8

    prompt = (
        "Which option is correct? Compute 2 + 2.\n"
        "A) 3\nB) 5\nC) 6\nD) 4\n"
        "Explain briefly, then end your final answer with exactly ANSWER: D."
    )
    extra_body: dict[str, object] = {"top_k": args.top_k, "repetition_penalty": args.repetition_penalty}
    if args.mode != "base":
        extra_body["chat_template_kwargs"] = {"enable_thinking": args.mode == "thinking"}

    client = OpenAI(api_key="EMPTY", base_url=args.base_url, timeout=600.0)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        seed=args.seed,
        extra_body=extra_body,
    )
    payload = response.model_dump(mode="json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    choice = response.choices[0]
    message = choice.message
    reasoning = getattr(message, "reasoning_content", None) or ""
    content = message.content or ""
    matches = ANSWER_RE.findall(content)
    result = {
        "output_path": str(args.output),
        "model": args.model,
        "mode": args.mode,
        "request": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "extra_body": extra_body,
        },
        "finish_reason": choice.finish_reason,
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "reasoning_present": bool(reasoning.strip()),
        "content_present": bool(content.strip()),
        "extracted_answer": matches[-1].upper() if matches else None,
        "content_tail": content[-500:],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not content.strip():
        raise SystemExit("final content is empty")
    if choice.finish_reason == "length":
        raise SystemExit("API smoke response was truncated at max_tokens")
    if args.mode == "thinking" and not reasoning.strip():
        raise SystemExit("thinking mode did not return reasoning_content")
    if args.mode == "non_thinking" and reasoning.strip():
        raise SystemExit("non-thinking mode unexpectedly returned reasoning_content")
    if not matches:
        raise SystemExit("final content does not contain a legal ANSWER: letter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
