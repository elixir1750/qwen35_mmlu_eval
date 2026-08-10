#!/usr/bin/env python3
"""Fixed-token throughput benchmark for SGLang and direct Transformers.

SGLang uses the OpenAI-compatible completions endpoint with token-ID prompts.
The Transformers path loads the checkpoint once and calls ``generate``.  The
two paths therefore share the exact input token IDs and requested output
length, while online concurrency is intentionally only claimed for SGLang
because ``transformers serve`` is not usable in this environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))], 6)


def stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": round(statistics.mean(values), 6) if values else None,
        "median": round(statistics.median(values), 6) if values else None,
        "p95": percentile(values, 0.95),
        "max": round(max(values), 6) if values else None,
    }


def gpu_snapshot() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        name, driver, used, total = [part.strip() for part in output.split(",", 3)]
        return {"name": name, "driver": driver, "memory_used_mib": int(float(used)), "memory_total_mib": int(float(total))}
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return {"name": None, "driver": None, "memory_used_mib": None, "memory_total_mib": None}


class GpuMonitor:
    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.max_used_mib = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            used = gpu_snapshot().get("memory_used_mib")
            if isinstance(used, int):
                self.max_used_mib = max(self.max_used_mib, used)
                self.samples += 1
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def model_revision(project: Path) -> str | None:
    info = project / "env/model_info.txt"
    if not info.exists():
        return None
    for line in info.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("hf_revision="):
            return line.split("=", 1)[1]
    return None


def prompt_ids(tokenizer: Any, length: int) -> list[int]:
    base = tokenizer.encode("The quick brown fox jumps over the lazy dog. ", add_special_tokens=False)
    if not base:
        raise RuntimeError("tokenizer produced an empty fixed prompt")
    return (base * ((length + len(base) - 1) // len(base)))[:length]


def output_count_from_text(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text or "", add_special_tokens=False))


def parse_completion_response(data: dict[str, Any], tokenizer: Any) -> tuple[int, str | None]:
    usage = data.get("usage") or {}
    count = usage.get("completion_tokens")
    choices = data.get("choices") or []
    text = choices[0].get("text", "") if choices and isinstance(choices[0], dict) else ""
    if not isinstance(count, int):
        count = output_count_from_text(tokenizer, text)
    return count, data.get("error")


async def sglang_request(
    client: Any,
    endpoint: str,
    model_name: str,
    ids: list[int],
    output_length: int,
    tokenizer: Any,
    stream: bool,
) -> dict[str, Any]:
    start_ns = time.monotonic_ns()
    first_token_ns: int | None = None
    output_text: list[str] = []
    output_tokens: int | None = None
    error: str | None = None
    try:
        payload = {
            "model": model_name,
            "prompt": ids,
            "max_tokens": output_length,
            "min_tokens": output_length,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": 42,
            "stream": stream,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
        }
        if stream:
            async with client.stream("POST", endpoint, json=payload) as response:
                if response.status_code != 200:
                    error = f"HTTP {response.status_code}: {(await response.aread())[:500].decode(errors='replace')}"
                else:
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if data.get("error"):
                            error = str(data["error"])
                        usage = data.get("usage") or {}
                        if isinstance(usage.get("completion_tokens"), int):
                            output_tokens = usage["completion_tokens"]
                        choices = data.get("choices") or []
                        text = choices[0].get("text", "") if choices and isinstance(choices[0], dict) else ""
                        if text:
                            first_token_ns = first_token_ns or time.monotonic_ns()
                            output_text.append(text)
        else:
            response = await client.post(endpoint, json=payload)
            if response.status_code != 200:
                error = f"HTTP {response.status_code}: {response.text[:500]}"
            else:
                data = response.json()
                output_tokens, error = parse_completion_response(data, tokenizer)
                choices = data.get("choices") or []
                text = choices[0].get("text", "") if choices and isinstance(choices[0], dict) else ""
                if text:
                    first_token_ns = time.monotonic_ns()
                    output_text.append(text)
    except Exception as exc:  # request errors are recorded per sample
        error = f"{type(exc).__name__}: {exc}"
    end_ns = time.monotonic_ns()
    if output_tokens is None:
        output_tokens = output_count_from_text(tokenizer, "".join(output_text))
    e2e_ms = (end_ns - start_ns) / 1_000_000
    ttft_ms = ((first_token_ns or end_ns) - start_ns) / 1_000_000
    tpot_ms = (e2e_ms - ttft_ms) / max(output_tokens - 1, 1)
    return {
        "success": error is None,
        "error": error,
        "input_tokens": len(ids),
        "output_tokens": output_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_latency_ms": e2e_ms,
    }


async def run_sglang_case(args: argparse.Namespace, tokenizer: Any, input_len: int, output_len: int, concurrency: int, layer: str) -> dict[str, Any]:
    import httpx

    ids = prompt_ids(tokenizer, input_len)
    endpoint = f"http://127.0.0.1:{args.port}/v1/completions"
    total_requests = args.repeats if layer == "single" else args.requests
    warmup = max(2, math.ceil(total_requests * 0.10))
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for _ in range(warmup):
            result = await sglang_request(client, endpoint, args.model_name, ids, output_len, tokenizer, stream=layer == "online")
            if not result["success"]:
                raise RuntimeError(f"SGLang warm-up failed: {result['error']}")
        start_ns = time.monotonic_ns()
        if layer == "single":
            results = [await sglang_request(client, endpoint, args.model_name, ids, output_len, tokenizer, stream=False) for _ in range(total_requests)]
        else:
            semaphore = asyncio.Semaphore(concurrency)

            async def bounded() -> dict[str, Any]:
                async with semaphore:
                    return await sglang_request(client, endpoint, args.model_name, ids, output_len, tokenizer, stream=True)

            results = await asyncio.gather(*(bounded() for _ in range(total_requests)))
        duration = (time.monotonic_ns() - start_ns) / 1_000_000_000
    return build_case(args, "sglang", layer, input_len, output_len, concurrency, total_requests, warmup, duration, results)


def build_case(args: argparse.Namespace, backend: str, layer: str, input_len: int, output_len: int, concurrency: int, requested: int, warmup: int, duration: float, results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in results if row["success"]]
    failed = [row for row in results if not row["success"]]
    actual_inputs = [row["input_tokens"] for row in completed]
    actual_outputs = [row["output_tokens"] for row in completed]
    same_length = bool(completed) and all(value == input_len for value in actual_inputs) and all(value == output_len for value in actual_outputs)
    output_total = sum(actual_outputs)
    input_total = sum(actual_inputs)
    return {
        "backend": backend,
        "layer": layer,
        "input_length_requested": input_len,
        "output_length_requested": output_len,
        "concurrency": concurrency,
        "requested_requests": requested,
        "warmup_requests": warmup,
        "completed_requests": len(completed),
        "failed_requests": len(failed),
        "duration_seconds": round(duration, 6),
        "request_per_second": round(len(completed) / duration, 6) if duration > 0 else None,
        "input_tokens_per_second": round(input_total / duration, 6) if duration > 0 else None,
        "output_tokens_per_second": round(output_total / duration, 6) if duration > 0 else None,
        "total_tokens_per_second": round((input_total + output_total) / duration, 6) if duration > 0 else None,
        "actual_input_tokens": stats([float(value) for value in actual_inputs]),
        "actual_output_tokens": stats([float(value) for value in actual_outputs]),
        "ttft_ms": stats([float(row["ttft_ms"]) for row in completed if isinstance(row.get("ttft_ms"), (int, float))]),
        "tpot_ms": stats([float(row["tpot_ms"]) for row in completed if isinstance(row.get("tpot_ms"), (int, float))]),
        "e2e_latency_ms": stats([float(row["e2e_latency_ms"]) for row in completed if isinstance(row.get("e2e_latency_ms"), (int, float))]),
        "same_input_output_length": same_length,
        "errors": failed[:10],
    }


def run_transformers(args: argparse.Namespace, tokenizer: Any, input_lengths: list[int], output_lengths: list[int]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForImageTextToText

    if not torch.cuda.is_available():
        raise RuntimeError("Transformers baseline requires a CUDA GPU")
    # Avoid requiring accelerate just to place one model on one GPU.  A plain
    # load followed by .to(cuda) is intentionally used for this baseline so
    # the repository remains runnable with the existing isolated environment.
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    model.to("cuda")
    model.eval()
    runs: list[dict[str, Any]] = []
    for input_len in input_lengths:
        ids = prompt_ids(tokenizer, input_len)
        inputs = {"input_ids": torch.tensor([ids], dtype=torch.long, device="cuda"), "attention_mask": torch.ones((1, input_len), dtype=torch.long, device="cuda")}
        for output_len in output_lengths:
            if args.layer == "online" and args.concurrency != 1:
                continue
            for _ in range(2):
                with torch.inference_mode():
                    model.generate(**inputs, max_new_tokens=output_len, min_new_tokens=output_len, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            results: list[dict[str, Any]] = []
            start_ns = time.monotonic_ns()
            with GpuMonitor() as monitor:
                for _ in range(args.repeats):
                    one_start = time.monotonic_ns()
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=output_len, min_new_tokens=output_len, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
                    torch.cuda.synchronize()
                    one_end = time.monotonic_ns()
                    actual = int(generated.shape[-1] - input_len)
                    results.append({
                        "success": actual == output_len,
                        "error": None if actual == output_len else f"output length {actual} != {output_len}",
                        "input_tokens": input_len,
                        "output_tokens": actual,
                        "ttft_ms": None,
                        "tpot_ms": None,
                        "e2e_latency_ms": (one_end - one_start) / 1_000_000,
                    })
            duration = (time.monotonic_ns() - start_ns) / 1_000_000_000
            case = build_case(args, "transformers", "single", input_len, output_len, 1, args.repeats, 2, duration, results)
            case["peak_gpu_memory_mib"] = monitor.max_used_mib
            runs.append(case)
    del model
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("sglang", "transformers"), required=True)
    parser.add_argument("--layer", choices=("single", "online"), default="single")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", dest="model_name", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--input-length", type=int, nargs="+", default=[256])
    parser.add_argument("--output-length", type=int, nargs="+", default=[64])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--cache-mode", choices=("disabled", "enabled"), default="enabled")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.model_path = args.model_path.resolve()
    args.model_name = args.model_name
    planned = {
        "backend": args.backend, "layer": args.layer, "model_path": str(args.model_path),
        "model_revision": model_revision(args.project_dir), "context_length": args.context_length,
        "input_lengths": args.input_length, "output_lengths": args.output_length,
        "concurrency": args.concurrency, "repeats": args.repeats, "requests": args.requests,
        "cache_mode": args.cache_mode, "mtp": False, "speculative_decoding": False,
    }
    if args.dry_run:
        print(json.dumps(planned, ensure_ascii=False, indent=2))
        return
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    start_ns = time.monotonic_ns()
    wall_start = now_iso()
    gpu_before = gpu_snapshot()
    with GpuMonitor() as monitor:
        if args.backend == "sglang":
            if args.layer == "single":
                runs = asyncio.run(run_all_sglang_single(args, tokenizer))
            else:
                runs = asyncio.run(run_all_sglang_online(args, tokenizer))
        else:
            runs = run_transformers(args, tokenizer, args.input_length, args.output_length)
        for run in runs:
            run["peak_gpu_memory_mib"] = monitor.max_used_mib
    gpu_after = gpu_snapshot()
    result = {
        "schema_version": 1,
        "kind": "throughput",
        "wall_clock_start": wall_start,
        "wall_clock_end": now_iso(),
        "monotonic_duration_seconds": round((time.monotonic_ns() - start_ns) / 1_000_000_000, 6),
        "hostname": platform.node(), "pid": os.getpid(),
        "gpu": gpu_after if gpu_after.get("name") else gpu_before,
        "backend": args.backend, "layer": args.layer,
        "backend_versions": {"sglang": package_version("sglang"), "transformers": package_version("transformers"), "torch": package_version("torch")},
        "model_path": str(args.model_path), "model_revision": model_revision(args.project_dir),
        "dtype": "bfloat16", "context_length": args.context_length, "cache_mode": args.cache_mode,
        "mtp": False, "speculative_decoding": False, "attention_backend": "sdpa" if args.backend == "transformers" else os.environ.get("SGLANG_ATTENTION_BACKEND", "server-default"),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "backend": args.backend, "runs": len(runs), "duration_seconds": result["monotonic_duration_seconds"]}, ensure_ascii=False))


async def run_all_sglang_single(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for input_len in args.input_length:
        for output_len in args.output_length:
            runs.append(await run_sglang_case(args, tokenizer, input_len, output_len, 1, "single"))
    return runs


async def run_all_sglang_online(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for input_len in args.input_length:
        for output_len in args.output_length:
            for concurrency in args.concurrency:
                args.requests = max(args.requests, max(20, 5 * concurrency)) if args.requests >= 20 else max(args.requests, max(8, 5 * concurrency))
                runs.append(await run_sglang_case(args, tokenizer, input_len, output_len, concurrency, "online"))
    return runs


if __name__ == "__main__":
    main()
