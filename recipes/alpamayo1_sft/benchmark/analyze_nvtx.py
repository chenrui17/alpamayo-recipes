#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize custom NVTX ranges from benchmark profiles (steps 5-7 capture)."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"

# Prefixes emitted by collate_nvtx.py, perf_utils wrapper, and trainer_nvtx.py.
RANGE_PREFIXES = (
    "get_batch_samples",
    "collate_fn",
    "collate_fn.inner",
    "collate.basic_collation",
    "collate.stack_tokenized",
    "collate.tokenizer",
    "collate.label_mask",
    "training_step",
)


def _load_nvtx_events(db_path: Path) -> list[tuple[int, int, int, str]]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT globalTid, start, end, text
        FROM NVTX_EVENTS
        WHERE text IS NOT NULL
        ORDER BY start
        """
    ).fetchall()
    con.close()
    return rows


def _match_suffix(text: str) -> str | None:
    for suffix in sorted(RANGE_PREFIXES, key=len, reverse=True):
        if text.endswith(suffix):
            return suffix
    return None


def _step_window_ns(rows: list[tuple], profile_start: int, profile_end: int) -> tuple[int, int] | None:
    """Time bounds [t0, t1] for optimizer steps [profile_start, profile_end] via training_step NVTX."""
    micro_per_step = 2  # CoC effective GA=2 on 8 GPUs
    start_idx = (profile_start - 1) * micro_per_step
    end_idx = profile_end * micro_per_step

    windows: list[tuple[int, int]] = []
    for rank in range(8):
        ts_events = sorted(
            (start, end)
            for _tid, start, end, text in rows
            if text == f"rank{rank}.training_step"
        )
        if len(ts_events) < end_idx:
            continue
        t0 = ts_events[start_idx][0]
        t1 = ts_events[end_idx - 1][1]
        windows.append((t0, t1))

    if not windows:
        return None
    return min(t0 for t0, _ in windows), max(t1 for _, t1 in windows)


def analyze_nvtx(name: str, profile_start: int = 5, profile_end: int = 7) -> dict:
    rep = PROFILE_DIR / f"{name}.nsys-rep"
    db = PROFILE_DIR / f"{name}.sqlite"
    if not db.exists() and rep.exists():
        import subprocess

        subprocess.run(
            ["nsys", "export", "--type=sqlite", "--output", str(db), str(rep)],
            check=True,
        )

    rows = _load_nvtx_events(db)
    steps_path = PROFILE_DIR / f"{name}_steps.json"
    steps_5_7 = []
    if steps_path.exists():
        steps_5_7 = json.loads(steps_path.read_text()).get("step_times_sec", [])[
            profile_start - 1 : profile_end
        ]

    window = _step_window_ns(rows, profile_start, profile_end)
    if window is not None:
        t0, t1 = window
        rows = [r for r in rows if r[1] >= t0 and r[2] <= t1]

    by_suffix: dict[str, list[float]] = defaultdict(list)
    by_rank_suffix: dict[tuple[int, str], list[float]] = defaultdict(list)

    for _tid, start, end, text in rows:
        suffix = _match_suffix(text)
        if suffix is None:
            continue
        dur = (end - start) / 1e9
        by_suffix[suffix].append(dur)
        if text.startswith("rank") and "." in text:
            rank_str = text.split(".", 1)[0]
            try:
                rank = int(rank_str.replace("rank", ""))
            except ValueError:
                continue
            by_rank_suffix[(rank, suffix)].append(dur)

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0, "total_sec": 0.0, "avg_ms": 0.0, "max_ms": 0.0}
        return {
            "n": len(values),
            "total_sec": sum(values),
            "avg_ms": 1000 * sum(values) / len(values),
            "max_ms": 1000 * max(values),
        }

    summary = {suffix: _stats(v) for suffix, v in sorted(by_suffix.items())}

    straggler: dict[str, dict] = {}
    for suffix in RANGE_PREFIXES:
        rank_totals = []
        for rank in range(8):
            vals = by_rank_suffix.get((rank, suffix), [])
            rank_totals.append((rank, sum(vals)))
        if not rank_totals:
            continue
        starts_by_rank: dict[int, list[float]] = defaultdict(list)
        for tid, start, end, text in rows:
            if not text.endswith(suffix):
                continue
            if not text.startswith("rank"):
                continue
            rank = int(text.split(".", 1)[0].replace("rank", ""))
            starts_by_rank[rank].append(start / 1e9)

        if starts_by_rank:
            first_starts = [min(v) for v in starts_by_rank.values() if v]
            if len(first_starts) >= 2:
                straggler[suffix] = {
                    "first_start_spread_ms": 1000 * (max(first_starts) - min(first_starts)),
                    "total_sec_by_rank": {r: round(t, 3) for r, t in rank_totals},
                }

    return {
        "name": name,
        "steps_5_7_sec": steps_5_7,
        "nvtx_window_sec": [round(window[0] / 1e9, 3), round(window[1] / 1e9, 3)] if window else None,
        "range_stats": summary,
        "straggler": straggler,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names",
        nargs="*",
        default=["coc_production_opt_nvtx"],
        help="Profile base name under benchmark/profiles/",
    )
    args = parser.parse_args()

    for name in args.names:
        result = analyze_nvtx(name)
        print(f"\n=== {name} ===")
        if result["steps_5_7_sec"]:
            avg = sum(result["steps_5_7_sec"]) / len(result["steps_5_7_sec"])
            print(f"steps 5-7: {[round(x, 3) for x in result['steps_5_7_sec']]} avg={avg:.3f}s")
        if result.get("nvtx_window_sec"):
            print(f"NVTX filter window (steps 5-7 via training_step): {result['nvtx_window_sec']} sec")
        print("\nNVTX range stats (steps 5-7 window, all ranks aggregated):")
        for suffix, st in result["range_stats"].items():
            print(
                f"  {suffix:28s} n={st['n']:4d}  "
                f"total={st['total_sec']:7.2f}s  avg={st['avg_ms']:7.1f}ms  max={st['max_ms']:7.1f}ms"
            )
        if result["straggler"]:
            print("\nRank straggler (first-start spread / total time by rank):")
            for suffix, info in result["straggler"].items():
                print(f"  {suffix}:")
                print(f"    first_start_spread_ms: {info['first_start_spread_ms']:.1f}")
                print(f"    total_sec_by_rank: {info['total_sec_by_rank']}")


if __name__ == "__main__":
    main()
