#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize per-rank dataloader timing JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _avg_phase(steps: dict, phase: str) -> float:
    vals = []
    for step_data in steps.values():
        phases = step_data.get("phases", {})
        if phase in phases and phases[phase].get("n", 0) > 0:
            vals.append(phases[phase]["avg_ms"])
    return sum(vals) / len(vals) if vals else 0.0


def _avg_derived(steps: dict, key: str) -> float:
    vals = [s.get("derived_ms", {}).get(key, 0.0) for s in steps.values()]
    vals = [v for v in vals if v > 0]
    return sum(vals) / len(vals) if vals else 0.0


def summarize_run(prefix: Path) -> None:
    files = sorted(prefix.parent.glob(f"{prefix.name}_rank*.json"))
    if not files:
        print(f"No timing files matching {prefix.name}_rank*.json")
        return

    print(f"=== {prefix.name} ({len(files)} ranks) ===")
    rank_derived: dict[str, list[float]] = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rank = payload.get("rank", path.stem.split("rank")[-1])
        steps = payload.get("steps", {})
        if not steps:
            continue
        derived = {
            "fetch": _avg_derived(steps, "fetch_total"),
            "worker_wait": _avg_derived(steps, "worker_wait_est"),
            "collate": _avg_derived(steps, "collate_total"),
            "gpu": _avg_derived(steps, "gpu_total"),
            "wall": _avg_derived(steps, "step_wall"),
            "other": _avg_derived(steps, "other_est"),
            "gpu_idle": _avg_derived(steps, "gpu_idle_serial_est"),
            "ds_load": _avg_derived(steps, "dataset_load_total"),
            "ds_pre": _avg_derived(steps, "dataset_preprocess_total"),
            "collate_tok": _avg_phase(steps, "collate.tokenizer"),
        }
        for k, v in derived.items():
            rank_derived.setdefault(k, []).append(v)
        print(
            f"rank {rank}: wall={derived['wall']:.0f}ms "
            f"fetch={derived['fetch']:.0f}ms "
            f"(load={derived['ds_load']:.0f} pre={derived['ds_pre']:.0f} "
            f"collate={derived['collate']:.0f} ipc~={derived['worker_wait']:.0f}) "
            f"gpu={derived['gpu']:.0f}ms opt/sync~={derived['other']:.0f}ms "
            f"gpu_idle~={derived['gpu_idle']:.0f}ms"
        )

    print("--- cross-rank avg ---")
    for key in [
        "wall",
        "fetch",
        "ds_load",
        "ds_pre",
        "collate",
        "worker_wait",
        "gpu",
        "other",
        "gpu_idle",
    ]:
        vals = rank_derived.get(key, [])
        if vals:
            print(f"  {key}: avg={sum(vals)/len(vals):.1f}ms max={max(vals):.1f}ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_name",
        nargs="?",
        default="coc_production_opt",
        help="Run name (looks for benchmark/profiles/<run>_dataloader_timing_rank*.json)",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    summarize_run(root / "benchmark" / "profiles" / f"{args.run_name}_dataloader_timing")


if __name__ == "__main__":
    main()
