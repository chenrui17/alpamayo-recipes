#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Nsight Systems profiles for production_opt configs (steps 5-7)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "benchmark" / "profiles"
NPROC = 8
MAX_STEPS = 7
PROFILE_START = 5
PROFILE_END = 7

COMMON_OVERRIDES = [
    f"benchmark.max_steps={MAX_STEPS}",
    "benchmark.stable_tail=3",
    "trainer.deepspeed=/raid/charlie/alpamayo/alpamayo-recipes/recipes/alpamayo1_sft/configs/deepspeed/zero2.json",
    "model.checkpoint_path=/raid/charlie/Alpamayo-R1-10B",
    "data.train_dataset.local_dir=/raid/charlie/pai_dataset",
    "data.val_dataset.local_dir=/raid/charlie/pai_dataset",
]

PRODUCTION_OPT = [
    "trainer.dataloader_num_workers=8",
    "trainer.dataloader_persistent_workers=true",
    "trainer.dataloader_prefetch_factor=4",
    "+benchmark.optimizations.zip_cache=true",
    "+benchmark.optimizations.collate_cache=true",
    "+benchmark.optimizations.tf32=true",
    "+benchmark.optimizations.cudnn_benchmark=true",
]

COC_COLLATE_ONLY = [
    "trainer.dataloader_num_workers=8",
    "trainer.dataloader_persistent_workers=true",
    "trainer.dataloader_prefetch_factor=4",
    "+benchmark.optimizations.zip_cache=false",
    "+benchmark.optimizations.collate_cache=true",
    "+benchmark.optimizations.tf32=false",
    "+benchmark.optimizations.cudnn_benchmark=false",
]

NVTX_DATALOADER = [
    "trainer.dataloader_num_workers=0",
    "trainer.dataloader_persistent_workers=false",
    "~trainer.dataloader_prefetch_factor",
]

TASK_CONFIG = {
    "no_coc_production_opt": "sft_bench_no_coc",
    "coc_production_opt": "sft_bench_coc",
    "coc_production_opt_nvtx": "sft_bench_coc",
}

TARGET_OVERRIDES: dict[str, list[str]] = {
    "no_coc_production_opt": PRODUCTION_OPT,
    "coc_production_opt": PRODUCTION_OPT,
    # CoC + collate_cache only; workers=0 so collate NVTX runs on the training rank.
    "coc_production_opt_nvtx": [
        "+benchmark.optimizations.zip_cache=false",
        "+benchmark.optimizations.collate_cache=true",
        "+benchmark.optimizations.tf32=false",
        "+benchmark.optimizations.cudnn_benchmark=false",
        *NVTX_DATALOADER,
    ],
}

NVTX_TARGETS = frozenset({"coc_production_opt_nvtx"})


def run_profile(name: str, force: bool = False) -> int:
    config_name = TASK_CONFIG[name]
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    output_base = PROFILE_DIR / name
    rep_path = Path(f"{output_base}.nsys-rep")
    log_path = PROFILE_DIR / f"{name}.log"
    timing_path = PROFILE_DIR / f"{name}_steps.json"

    if rep_path.exists() and not force:
        print(f">>> Skip {name}: {rep_path} exists (use --force to rerun)", flush=True)
        return 0

    overrides = COMMON_OVERRIDES + TARGET_OVERRIDES.get(name, PRODUCTION_OPT) + [
        f"benchmark.run_name={name}_profile",
        f"benchmark.result_path=benchmark/profiles/{name}_steps.json",
        f"paths.output_dir=benchmark/profiles/runs/{name}",
    ]
    nsys_extra: list[str] = []
    if name in NVTX_TARGETS:
        overrides.append("+benchmark.nvtx_markers=true")
        # Full trace: cudaProfilerApi window misses CPU collate on the timeline for
        # steps 5-7 (prefetch + capture timing). Zoom to steps 5-7 in nsys-ui instead.
    else:
        overrides.extend(
            [
                f"+benchmark.profile_start={PROFILE_START}",
                f"+benchmark.profile_end={PROFILE_END}",
            ]
        )
        nsys_extra = ["--capture-range=cudaProfilerApi", "--capture-range-end=stop"]

    env = os.environ.copy()
    env.setdefault("NSYS_NVTX_PROFILER_REGISTER_ONLY", "0")
    master_port = env.get("MASTER_PORT", "29502")

    cmd = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt,cudnn,cublas",
        "--sample=none",
        "--cpuctxsw=none",
        *nsys_extra,
        f"--output={output_base}",
        "--force-overwrite=true",
        "torchrun",
        f"--nproc_per_node={NPROC}",
        f"--master_port={master_port}",
        "-m",
        "alpamayo1_sft.bench_hf",
        "--config-path",
        "pkg://alpamayo1_sft/configs",
        "--config-name",
        config_name,
        *overrides,
    ]

    print(f"\n>>> Profiling {name} (steps {PROFILE_START}-{PROFILE_END})", flush=True)
    print(" ".join(cmd), flush=True)

    with log_path.open("w") as log_f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT, env=env)

    if proc.returncode != 0:
        print(f">>> FAILED {name}, see {log_path}", flush=True)
        return proc.returncode

    if timing_path.exists():
        with timing_path.open() as f:
            data = json.load(f)
        steps = data.get("step_times_sec", [])
        if len(steps) >= PROFILE_END:
            avg = sum(steps[PROFILE_START - 1 : PROFILE_END]) / (PROFILE_END - PROFILE_START + 1)
            print(f">>> Done {name}: steps 5-7 avg={avg:.3f}s -> {rep_path}", flush=True)
        else:
            print(f">>> Done {name} -> {rep_path}", flush=True)
    else:
        print(f">>> Done {name} -> {rep_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        choices=list(TASK_CONFIG.keys()),
        help="Profile targets (default: both)",
    )
    parser.add_argument("--force", action="store_true", help="Rerun even if .nsys-rep exists")
    args = parser.parse_args()
    targets = args.targets or list(TASK_CONFIG.keys())

    rc = 0
    for name in targets:
        rc |= run_profile(name, force=args.force)
    return rc


if __name__ == "__main__":
    sys.exit(main())
