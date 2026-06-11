#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize Nsight Systems profiles for production_opt runs (steps 5-7)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
COPY = {1: "H2D", 2: "D2H", 8: "D2D"}


def _cat(cur, pattern: str) -> tuple[int, int]:
    row = cur.execute(
        f"""
        SELECT SUM(k.end-k.start), COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s ON k.demangledName = s.id
        LEFT JOIN StringIds s2 ON k.shortName = s2.id
        WHERE COALESCE(s.value, s2.value, '') LIKE '%{pattern}%'
        """
    ).fetchone()
    return row[0] or 0, row[1] or 0


def analyze(name: str) -> dict:
    steps = json.loads((PROFILE_DIR / f"{name}_steps.json").read_text())
    s5_7 = steps["step_times_sec"][4:7]
    db = sqlite3.connect(PROFILE_DIR / f"{name}.sqlite")
    cur = db.cursor()

    total_kern = cur.execute("SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] or 0

    categories = {
        "nccl_allreduce": _cat(cur, "ncclDevKernel_AllReduce"),
        "nccl_allgather": _cat(cur, "ncclDevKernel_AllGather"),
        "flash_attn": _cat(cur, "flash::"),
        "nvjet_gemm": _cat(cur, "nvjet"),
        "cutlass_gemm": _cat(cur, "cutlass"),
        "fused_adam": _cat(cur, "FusedAdam"),
    }

    api = cur.execute(
        """
        SELECT s.value, SUM(r.end-r.start), COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        GROUP BY s.value ORDER BY 2 DESC LIMIT 10
        """
    ).fetchall()

    memcpy_rows = cur.execute(
        "SELECT copyKind, SUM(end-start), SUM(bytes), COUNT(*) "
        "FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
    ).fetchall()
    memcpy = {
        COPY.get(r[0], str(r[0])): {"sec": r[1] / 1e9, "gb": r[2] / 1e9, "ops": r[3]}
        for r in memcpy_rows
    }

    sync = cur.execute(
        "SELECT SUM(end-start), COUNT(*) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
    ).fetchone()

    db.close()

    wall = sum(s5_7)
    return {
        "name": name,
        "steps_5_7_sec": s5_7,
        "avg_step_sec": sum(s5_7) / 3,
        "wall_3step_sec": wall,
        "gpu_kernel_total_sec": total_kern / 1e9,
        "categories": {k: {"sec": v[0] / 1e9, "pct": 100 * v[0] / total_kern, "n": v[1]} for k, v in categories.items() if v[0]},
        "memcpy": memcpy,
        "api_top": [(r[0], r[1] / 1e9, r[2]) for r in api],
        "gpu_sync_sec": (sync[0] or 0) / 1e9,
        "gpu_sync_events": sync[1] or 0,
    }


def main() -> None:
    for name in ("no_coc_production_opt", "coc_production_opt"):
        r = analyze(name)
        print(f"\n=== {name} ===")
        print(f"steps 5-7: {[round(x, 3) for x in r['steps_5_7_sec']]}")
        print(f"avg step: {r['avg_step_sec']:.3f}s")
        print(f"GPU kernels (8-GPU sum): {r['gpu_kernel_total_sec']:.1f}s")
        for k, v in r["categories"].items():
            print(f"  {k}: {v['sec']:.1f}s ({v['pct']:.1f}%)")
        print("memcpy:", r["memcpy"])
        print("top API:", r["api_top"][:4])
        print(f"sync wait: {r['gpu_sync_sec']:.1f}s")


if __name__ == "__main__":
    main()
