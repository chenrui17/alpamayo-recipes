# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze benchmark JSON using only the Python standard library."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


def statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("No valid step intervals in the requested window")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Step intervals must be positive finite seconds")
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.9
    lower, upper = math.floor(position), math.ceil(position)
    p90 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "count": len(values), "mean_seconds": mean(values), "median_seconds": median(values),
        "p90_seconds": p90, "min_seconds": min(values), "max_seconds": max(values),
        "total_seconds": sum(values),
    }


def summarize(
    result: dict[str, Any],
    global_samples_per_step: float | None = None,
    global_tokens_per_step: float | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    if result.get("schema_version") != 1:
        raise ValueError("Unsupported or missing benchmark schema_version")
    if not allow_incomplete and not result.get("complete"):
        raise ValueError("Benchmark did not complete its requested measurement window")
    start, end = int(result["steady_start_step"]), int(result["steady_end_step"])
    expected = set(range(start, end + 1))
    ranks = result["ranks"]
    if not ranks or len(ranks) != int(result["world_size"]):
        raise ValueError("Rank payload count does not match world_size")
    if {int(rank["rank"]) for rank in ranks} != set(range(len(ranks))):
        raise ValueError("Duplicate or missing rank payloads")

    def window(records: list[dict[str, Any]]) -> dict[int, float]:
        values: dict[int, float] = {}
        for record in records:
            step = int(record["step"])
            if start <= step <= end and record.get("step_seconds") is not None:
                if step in values:
                    raise ValueError(f"Duplicate optimizer step {step}")
                values[step] = float(record["step_seconds"])
        if not allow_incomplete and set(values) != expected:
            raise ValueError("Missing optimizer-step intervals in the measurement window")
        return values

    sample_count = global_samples_per_step if global_samples_per_step is not None else result.get("global_samples_per_step")
    token_count = global_tokens_per_step if global_tokens_per_step is not None else result.get("global_tokens_per_step")
    for count in (sample_count, token_count):
        if count is not None and (not math.isfinite(count) or count <= 0):
            raise ValueError("Throughput counts must be positive finite values")

    def with_throughput(stats: dict[str, Any], divisor: int = 1) -> dict[str, Any]:
        if sample_count is not None:
            stats["samples_per_second"] = sample_count / divisor / stats["mean_seconds"]
        if token_count is not None:
            stats["tokens_per_second"] = token_count / divisor / stats["mean_seconds"]
        return stats

    rank_values = [window(rank["records"]) for rank in ranks]
    common_steps = set.intersection(*(set(values) for values in rank_values))
    rank_max = {step: max(values[step] for values in rank_values) for step in sorted(common_steps)}
    # Recompute rank maxima from raw ranks, rather than trusting a cached summary.
    summary = {
        "run_id": result["run_id"], "mode": result["mode"], "complete": result["complete"],
        "world_size": len(ranks), "steady_window": [start, end],
        "p90_method": "linear interpolation at (n - 1) * 0.9",
        "global_samples_per_step": sample_count, "global_tokens_per_step": token_count,
        "rank_max": with_throughput(statistics(list(rank_max.values()))),
        "rank_max_records": [{"step": step, "step_seconds": seconds} for step, seconds in rank_max.items()],
        "per_rank": [],
    }
    for rank, values in zip(ranks, rank_values):
        sync_values = [
            float(record["sync_seconds"]) for record in rank["records"]
            if int(record["step"]) in values
        ]
        summary["per_rank"].append({
            "rank": rank["rank"],
            **with_throughput(statistics(list(values.values())), divisor=len(ranks)),
            "mean_sync_seconds": mean(sync_values),
            "max_memory_allocated_bytes": rank.get("max_memory_allocated_bytes"),
            "max_memory_reserved_bytes": rank.get("max_memory_reserved_bytes"),
            "loss_logs": [entry for entry in rank.get("logs", []) if "loss" in entry],
        })
    for name in ("max_memory_allocated_bytes", "max_memory_reserved_bytes"):
        memory = [rank[name] for rank in ranks if rank.get(name) is not None]
        summary[name] = max(memory) if memory else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--global-samples-per-step", type=float)
    parser.add_argument("--global-tokens-per-step", type=float)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    summaries = [summarize(json.loads(path.read_text()), args.global_samples_per_step,
                           args.global_tokens_per_step, args.allow_incomplete) for path in args.results]
    output = json.dumps(summaries[0] if len(summaries) == 1 else summaries, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
