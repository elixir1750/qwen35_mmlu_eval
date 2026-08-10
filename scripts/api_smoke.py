#!/usr/bin/env python3
import json
import os
from pathlib import Path

from openai import OpenAI


base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
output_path = Path(os.environ.get("API_SMOKE_OUTPUT", "outputs/smoke/api_smoke.json"))
output_path.parent.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=600.0)
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=[
        {
            "role": "user",
            "content": (
                "Which option is correct? Compute 2 + 2.\n"
                "A) 3\nB) 5\nC) 6\nD) 4\n"
                "Think step by step and end your final line with ANSWER: D."
            ),
        }
    ],
    max_tokens=32768,
    temperature=1.0,
    top_p=0.95,
    presence_penalty=1.5,
    seed=42,
    extra_body={"top_k": 20, "repetition_penalty": 1.0},
)

payload = response.model_dump(mode="json")
output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

choice = response.choices[0]
message = choice.message
reasoning = getattr(message, "reasoning_content", None)
content = message.content or ""
finish_reason = choice.finish_reason
print(json.dumps({
    "output_path": str(output_path),
    "finish_reason": finish_reason,
    "reasoning_chars": len(reasoning or ""),
    "content_chars": len(content),
    "reasoning_present": bool(reasoning),
    "content_present": bool(content),
    "content_tail": content[-300:],
}, ensure_ascii=False, indent=2))

if not reasoning:
    raise SystemExit("reasoning_content is empty; reasoning parser is not verified")
if not content.strip():
    raise SystemExit("final content is empty; reasoning parser may have swallowed final answer")
if finish_reason == "length":
    raise SystemExit("API smoke response was truncated at max_tokens")
