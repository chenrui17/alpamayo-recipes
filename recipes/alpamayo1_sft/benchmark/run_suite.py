#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Stage-1 throughput benchmarks for CoC disabled / enabled tasks."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmark" / "results"
RESULTS_MD = ROOT / "benchmark" / "RESULTS.md"
NPROC = 8
MAX_STEPS = 50
STABLE_TAIL = 40

BASE_OVERRIDES = [
    f"benchmark.max_steps={MAX_STEPS}",
    f"benchmark.stable_tail={STABLE_TAIL}",
    "trainer.deepspeed=/raid/charlie/alpamayo/alpamayo-recipes/recipes/alpamayo1_sft/configs/deepspeed/zero2.json",
    "model.checkpoint_path=/raid/charlie/Alpamayo-R1-10B",
    "data.train_dataset.local_dir=/raid/charlie/pai_dataset",
    "data.val_dataset.local_dir=/raid/charlie/pai_dataset",
]

TASK_CONFIG = {
    "no_coc": "sft_bench_no_coc",
    "coc": "sft_bench_coc",
}


def _opt(**kwargs) -> list[str]:
    return [f"+benchmark.optimizations.{k}={'true' if v else 'false'}" for k, v in kwargs.items()]


def _gpu_compute_pids() -> list[str]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _ensure_gpu_idle() -> None:
    pids = _gpu_compute_pids()
    if pids:
        print(
            f">>> ABORT: GPU already in use by PIDs {', '.join(pids[:8])}"
            f"{'...' if len(pids) > 8 else ''}. "
            "Kill stale benchmark jobs before re-running:\n"
            "  pkill -9 -f 'alpamayo1_sft.bench_hf'\n"
            "  pkill -9 -f 'benchmark/run_suite.py'",
            flush=True,
        )
        sys.exit(1)


def _find_free_port(start: int = 29500, end: int = 29600) -> int:
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free TCP port in {start}-{end - 1}")


EXPERIMENTS: dict[str, list[dict]] = {
    "no_coc": [
        {
            "name": "no_coc_baseline",
            "desc": "workers=2, no pipeline opts (pre-optimization baseline)",
            "extra": [
                "trainer.dataloader_num_workers=2",
                "trainer.dataloader_persistent_workers=false",
                *_opt(zip_cache=False, collate_cache=False, tf32=False, cudnn_benchmark=False),
            ],
        },
        {
            "name": "no_coc_dl_workers8",
            "desc": "DataLoader parallelism only",
            "extra": [
                "trainer.dataloader_num_workers=8",
                "trainer.dataloader_persistent_workers=true",
                "trainer.dataloader_prefetch_factor=4",
                *_opt(zip_cache=False, collate_cache=False, tf32=False, cudnn_benchmark=False),
            ],
        },
        {
            "name": "no_coc_production_opt",
            "desc": "Production-safe stack: workers + zip + collate + TF32",
            "extra": [
                "trainer.dataloader_num_workers=8",
                "trainer.dataloader_persistent_workers=true",
                "trainer.dataloader_prefetch_factor=4",
                *_opt(zip_cache=True, collate_cache=True, tf32=True, cudnn_benchmark=True),
            ],
        },
    ],
    "coc": [
        {
            "name": "coc_baseline",
            "desc": "workers=2, no pipeline opts",
            "extra": [
                "trainer.dataloader_num_workers=2",
                "trainer.dataloader_persistent_workers=false",
                *_opt(zip_cache=False, collate_cache=False, tf32=False, cudnn_benchmark=False),
            ],
        },
        {
            "name": "coc_dl_workers8",
            "desc": "DataLoader parallelism only",
            "extra": [
                "trainer.dataloader_num_workers=8",
                "trainer.dataloader_persistent_workers=true",
                "trainer.dataloader_prefetch_factor=4",
                *_opt(zip_cache=False, collate_cache=False, tf32=False, cudnn_benchmark=False),
            ],
        },
        {
            "name": "coc_production_opt",
            "desc": "Production-safe stack: workers + zip + collate + TF32",
            "extra": [
                "trainer.dataloader_num_workers=8",
                "trainer.dataloader_persistent_workers=true",
                "trainer.dataloader_prefetch_factor=4",
                *_opt(zip_cache=True, collate_cache=True, tf32=True, cudnn_benchmark=True),
            ],
        },
        {
            "name": "coc_production_opt_workers2",
            "desc": "Production stack with workers=2 (less CPU contention)",
            "extra": [
                "trainer.dataloader_num_workers=2",
                "trainer.dataloader_persistent_workers=true",
                "trainer.dataloader_prefetch_factor=4",
                *_opt(zip_cache=True, collate_cache=True, tf32=True, cudnn_benchmark=True),
            ],
        },
        {
            "name": "coc_production_opt_preprocess",
            "desc": "production_opt + preprocess_cache (clip-level VLA preprocess)",
            "extra": [
                "trainer.dataloader_num_workers=8",
                "trainer.dataloader_persistent_workers=true",
                "trainer.dataloader_prefetch_factor=4",
                *_opt(
                    zip_cache=True,
                    collate_cache=True,
                    tf32=True,
                    cudnn_benchmark=True,
                    preprocess_cache=True,
                ),
            ],
        },
        {
            "name": "coc_production_opt_loadcache",
            "desc": "collate + load_cache (full sample) + TF32, zip off for CoC",
            "extra": [
                "trainer.dataloader_num_workers=8",
                "trainer.dataloader_persistent_workers=true",
                "trainer.dataloader_prefetch_factor=4",
                *_opt(
                    zip_cache=False,
                    collate_cache=True,
                    tf32=True,
                    cudnn_benchmark=True,
                    load_cache=True,
                ),
            ],
        },
        {
            "name": "coc_production_opt_full_cache",
            "desc": "collate + preprocess + load_cache + TF32",
            "extra": [
                "trainer.dataloader_num_workers=8",
                "trainer.dataloader_persistent_workers=true",
                "trainer.dataloader_prefetch_factor=4",
                *_opt(
                    zip_cache=False,
                    collate_cache=True,
                    tf32=True,
                    cudnn_benchmark=True,
                    preprocess_cache=True,
                    load_cache=True,
                ),
            ],
        },
        {
            "name": "coc_dl_workers8_loadcache",
            "desc": "workers=8 + load_cache only",
            "extra": [
                "trainer.dataloader_num_workers=8",
                "trainer.dataloader_persistent_workers=true",
                "trainer.dataloader_prefetch_factor=4",
                *_opt(
                    zip_cache=False,
                    collate_cache=False,
                    tf32=False,
                    cudnn_benchmark=False,
                    load_cache=True,
                ),
            ],
        },
        {
            "name": "coc_dl_workers8_preprocess",
            "desc": "workers=8 + preprocess_cache only",
            "extra": [
                "trainer.dataloader_num_workers=8",
                "trainer.dataloader_persistent_workers=true",
                "trainer.dataloader_prefetch_factor=4",
                *_opt(
                    zip_cache=False,
                    collate_cache=False,
                    tf32=False,
                    cudnn_benchmark=False,
                    preprocess_cache=True,
                ),
            ],
        },
        {
            "name": "coc_dl_workers8_zip",
            "desc": "workers=8 + zip_cache only",
            "extra": [
                "trainer.dataloader_num_workers=8",
                "trainer.dataloader_persistent_workers=true",
                "trainer.dataloader_prefetch_factor=4",
                *_opt(zip_cache=True, collate_cache=False, tf32=False, cudnn_benchmark=False),
            ],
        },
        {
            "name": "coc_dl_workers8_collate",
            "desc": "workers=8 + collate_cache only",
            "extra": [
                "trainer.dataloader_num_workers=8",
                "trainer.dataloader_persistent_workers=true",
                "trainer.dataloader_prefetch_factor=4",
                *_opt(zip_cache=False, collate_cache=True, tf32=False, cudnn_benchmark=False),
            ],
        },
        {
            "name": "coc_dl_workers8_tf32",
            "desc": "workers=8 + tf32 only",
            "extra": [
                "trainer.dataloader_num_workers=8",
                "trainer.dataloader_persistent_workers=true",
                "trainer.dataloader_prefetch_factor=4",
                *_opt(zip_cache=False, collate_cache=False, tf32=True, cudnn_benchmark=False),
            ],
        },
        {
            "name": "coc_dl_workers8_cudnn",
            "desc": "workers=8 + cudnn_benchmark only",
            "extra": [
                "trainer.dataloader_num_workers=8",
                "trainer.dataloader_persistent_workers=true",
                "trainer.dataloader_prefetch_factor=4",
                *_opt(zip_cache=False, collate_cache=False, tf32=False, cudnn_benchmark=True),
            ],
        },
    ],
}

COC_W8_ABLATION = {
    "coc_dl_workers8_zip",
    "coc_dl_workers8_collate",
    "coc_dl_workers8_tf32",
    "coc_dl_workers8_cudnn",
}


def run_one(task: str, exp: dict) -> dict:
    name = exp["name"]
    config_name = TASK_CONFIG[task]
    result_path = RESULTS_DIR / f"{name}.json"
    log_path = RESULTS_DIR / f"{name}.log"

    if result_path.exists() and result_path.stat().st_size > 200:
        print(f">>> Skip {name}: {result_path} already exists", flush=True)
        with result_path.open() as f:
            data = json.load(f)
        return {
            "name": name,
            "task": task,
            "status": "ok",
            "avg": data["avg_step_sec_stable_tail"],
            "avg_all": data["avg_step_sec_all"],
            "workers": data.get("num_workers"),
            "optimizations": data.get("config", {}).get("optimizations", {}),
            "skipped": True,
        }

    overrides = BASE_OVERRIDES + [
        f"benchmark.run_name={name}",
        f"benchmark.result_path=benchmark/results/{name}.json",
        f"paths.output_dir=benchmark/runs/{name}",
    ] + exp["extra"]

    _ensure_gpu_idle()
    env = os.environ.copy()
    master_port = _find_free_port()

    cmd = [
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
    print(f"\n>>> Running {name}: {exp['desc']} (master_port={master_port})", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        print(f">>> FAILED {name}, see {log_path}", flush=True)
        return {"name": name, "task": task, "status": "failed", "log": str(log_path)}
    with result_path.open() as f:
        data = json.load(f)
    print(f">>> Done {name}: avg={data['avg_step_sec_stable_tail']:.3f}s", flush=True)
    return {
        "name": name,
        "task": task,
        "status": "ok",
        "avg": data["avg_step_sec_stable_tail"],
        "avg_all": data["avg_step_sec_all"],
        "workers": data.get("num_workers"),
        "optimizations": data.get("config", {}).get("optimizations", {}),
    }


def _fmt_opt(opt: dict) -> str:
    keys = [
        ("zip_cache", "zip"),
        ("collate_cache", "collate"),
        ("preprocess_cache", "preprocess"),
        ("load_cache", "load"),
        ("tf32", "tf32"),
        ("cudnn_benchmark", "cudnn"),
    ]
    enabled = [label for k, label in keys if opt.get(k)]
    return ",".join(enabled) if enabled else "-"


def _load_result_row(name: str, task: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists() or path.stat().st_size <= 200:
        return None
    with path.open() as f:
        data = json.load(f)
    return {
        "name": name,
        "task": task,
        "status": "ok",
        "avg": data["avg_step_sec_stable_tail"],
        "avg_all": data["avg_step_sec_all"],
        "workers": data.get("num_workers"),
        "optimizations": data.get("config", {}).get("optimizations", {}),
    }


def _collect_all_rows(fresh: list[dict]) -> list[dict]:
    by_name = {r["name"]: r for r in fresh}
    for task, exps in EXPERIMENTS.items():
        for exp in exps:
            name = exp["name"]
            if name in by_name:
                continue
            loaded = _load_result_row(name, task)
            if loaded:
                by_name[name] = loaded
    order = [exp["name"] for exps in EXPERIMENTS.values() for exp in exps]
    return [by_name[n] for n in order if n in by_name]


def write_results(rows: list[dict]) -> None:
    rows = _collect_all_rows(rows)
    lines = [
        "# Alpamayo-1 SFT Stage-1 Throughput Benchmark",
        "",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        f"Hardware: {NPROC}x H20, single node",
        f"Method: {MAX_STEPS} optimizer steps, average of last {STABLE_TAIL} stable steps",
        f"Tasks: CoC disabled (`sft_bench_no_coc`) / CoC enabled (`sft_bench_coc`)",
        "",
        "| Run | Task | Avg Step (s) | vs Task Baseline | workers | Optimizations |",
        "|-----|------|-------------:|-----------------:|--------:|---------------|",
    ]
    baselines: dict[str, float] = {}
    for r in rows:
        if r["status"] == "ok" and r["name"].endswith("_baseline"):
            baselines[r["task"]] = r["avg"]

    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['name']} | {r['task']} | FAILED | — | — | see {r.get('log', '')} |")
            continue
        base = baselines.get(r["task"])
        gain = f"{(base / r['avg'] - 1) * 100:+.1f}%" if base else "—"
        lines.append(
            f"| {r['name']} | {r['task']} | {r['avg']:.3f} | {gain} | {r['workers']} | {_fmt_opt(r['optimizations'])} |"
        )

    w8_ref = next((r for r in rows if r["name"] == "coc_dl_workers8" and r["status"] == "ok"), None)
    ablation_rows = [r for r in rows if r["name"] in COC_W8_ABLATION]
    if w8_ref and ablation_rows:
        lines.extend([
            "",
            "## CoC workers=8 single-optimization ablation",
            "",
            f"Reference: `coc_dl_workers8` = **{w8_ref['avg']:.3f} s** (workers=8, no zip/collate/tf32/cudnn)",
            "",
            "| Run | Avg Step (s) | vs coc_dl_workers8 | Enabled |",
            "|-----|-------------:|-------------------:|---------|",
        ])
        ref_avg = w8_ref["avg"]
        for r in ablation_rows:
            if r["status"] != "ok":
                lines.append(f"| {r['name']} | FAILED | — | see {r.get('log', '')} |")
                continue
            gain = f"{(ref_avg / r['avg'] - 1) * 100:+.1f}%"
            lines.append(
                f"| {r['name']} | {r['avg']:.3f} | {gain} | {_fmt_opt(r['optimizations'])} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "详细分析见 [`benchmark/PROFILE_ANALYSIS.md`](PROFILE_ANALYSIS.md)。",
    ])

    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {RESULTS_MD}", flush=True)


def main() -> int:
    args = sys.argv[1:]
    if "--check-gpu-only" in args:
        _ensure_gpu_idle()
        print(">>> GPUs idle, OK to run benchmarks", flush=True)
        return 0

    exp_filter: set[str] | None = None
    args = [a for a in args if a != "--check-gpu-only"]
    if not args:
        tasks = list(EXPERIMENTS.keys())
    elif all(a in EXPERIMENTS for a in args):
        tasks = args
    else:
        tasks = list(EXPERIMENTS.keys())
        exp_filter = set(args)

    rows: list[dict] = []
    for task in tasks:
        for exp in EXPERIMENTS[task]:
            if exp_filter and exp["name"] not in exp_filter:
                continue
            rows.append(run_one(task, exp))

    write_results(rows)
    failed = [r for r in rows if r["status"] != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
