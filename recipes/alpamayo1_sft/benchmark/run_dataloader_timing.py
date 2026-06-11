#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run dataloader timing on CoC benchmark configs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NPROC = 8
MAX_STEPS = 12
TIMING_START = 5

PROFILES: dict[str, list[str]] = {
    "coc_production_opt": [
        "benchmark.run_name=coc_production_opt",
        "+benchmark.dataloader_timing_path=benchmark/profiles/coc_production_opt_dataloader_timing.json",
        "trainer.dataloader_num_workers=8",
        "trainer.dataloader_prefetch_factor=4",
        "+benchmark.optimizations.zip_cache=true",
        "+benchmark.optimizations.collate_cache=true",
        "+benchmark.optimizations.tf32=true",
        "+benchmark.optimizations.cudnn_benchmark=true",
    ],
    "coc_production_opt_preprocess": [
        "benchmark.run_name=coc_production_opt_preprocess",
        "+benchmark.dataloader_timing_path=benchmark/profiles/coc_production_opt_preprocess_dataloader_timing.json",
        "trainer.dataloader_num_workers=8",
        "trainer.dataloader_prefetch_factor=4",
        "+benchmark.optimizations.zip_cache=true",
        "+benchmark.optimizations.collate_cache=true",
        "+benchmark.optimizations.preprocess_cache=true",
        "+benchmark.optimizations.tf32=true",
        "+benchmark.optimizations.cudnn_benchmark=true",
    ],
}

COMMON = [
    "trainer.deepspeed=/raid/charlie/alpamayo/alpamayo-recipes/recipes/alpamayo1_sft/configs/deepspeed/zero2.json",
    "model.checkpoint_path=/raid/charlie/Alpamayo-R1-10B",
    "data.train_dataset.local_dir=/raid/charlie/pai_dataset",
    "data.val_dataset.local_dir=/raid/charlie/pai_dataset",
    "trainer.dataloader_persistent_workers=true",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="coc_production_opt_preprocess",
        choices=sorted(PROFILES.keys()),
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--timing-start", type=int, default=TIMING_START)
    args = parser.parse_args()

    overrides = [
        f"benchmark.max_steps={args.max_steps}",
        "benchmark.stable_tail=5",
        "+benchmark.dataloader_timing=true",
        f"+benchmark.dataloader_timing_start={args.timing_start}",
        *COMMON,
        *PROFILES[args.profile],
    ]

    cmd = [
        "torchrun",
        f"--nproc_per_node={NPROC}",
        "-m",
        "alpamayo1_sft.bench_hf",
        "--config-path",
        "pkg://alpamayo1_sft/configs",
        "--config-name",
        "sft_bench_coc",
        *overrides,
    ]
    print(" ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("MASTER_PORT", "29504")
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    if proc.returncode != 0:
        return proc.returncode

    analyze = [
        sys.executable,
        str(ROOT / "benchmark" / "analyze_dataloader_timing.py"),
        args.profile,
    ]
    return subprocess.run(analyze, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
