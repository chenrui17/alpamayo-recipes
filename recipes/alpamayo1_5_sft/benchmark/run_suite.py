#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run throughput benchmark matrix and append results to benchmark/RESULTS.md."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmark" / "results"
RESULTS_JSONL = RESULTS_DIR / "history.jsonl"
RESULTS_MD = ROOT / "benchmark" / "RESULTS.md"
VENV = ROOT / "a1_5_sft" / "bin" / "activate"
NPROC = 8
TARGET_THROUGHPUT_GAIN = 0.80  # +80% throughput => avg_step *= 1/(1+0.8)
FULL_SUITE = os.environ.get("BENCHMARK_FULL", "0") == "1"


TRAINER_OVERRIDE_KEYS = {
    "dataloader_num_workers",
    "dataloader_pin_memory",
    "dataloader_persistent_workers",
    "dataloader_prefetch_factor",
    "torch_compile",
    "torch_compile_backend",
    "torch_compile_mode",
    "gradient_checkpointing",
    "gradient_accumulation_steps",
}


def hydra_overrides(experiment: dict) -> list[str]:
    o = [
        f"benchmark.run_name={experiment['name']}",
        f"benchmark.result_path=benchmark/results/{experiment['name']}.json",
    ]
    bench = experiment.get("benchmark") or {}
    opt = bench.get("optimizations") or {}
    for k, v in opt.items():
        if isinstance(v, bool):
            o.append(f"benchmark.optimizations.{k}={'true' if v else 'false'}")
        else:
            o.append(f"benchmark.optimizations.{k}={v}")
    for k, v in bench.items():
        if k in ("optimizations", "run_name", "result_path", "max_steps", "stable_tail"):
            continue
        if k in TRAINER_OVERRIDE_KEYS:
            key_path = f"trainer.{k}"
        else:
            key_path = f"+benchmark.{k}"
        if isinstance(v, bool):
            o.append(f"{key_path}={'true' if v else 'false'}")
        else:
            o.append(f"{key_path}={v}")
    data = experiment.get("data") or {}
    for path, val in _flatten("data", data):
        if isinstance(val, bool):
            o.append(f"{path}={'true' if val else 'false'}")
        elif isinstance(val, list):
            o.append(f"{path}=[{','.join(str(x) for x in val)}]")
        else:
            o.append(f"{path}={val}")
    trainer_extra = experiment.get("trainer") or {}
    for k, v in trainer_extra.items():
        if isinstance(v, bool):
            o.append(f"trainer.{k}={'true' if v else 'false'}")
        else:
            o.append(f"trainer.{k}={v}")
    return o


def _flatten(prefix: str, obj: dict):
    for k, v in obj.items():
        p = f"{prefix}.{k}"
        if isinstance(v, dict):
            yield from _flatten(p, v)
        else:
            yield p, v


EXPERIMENTS: list[dict] = [
    {
        "name": "baseline",
        "description": "Current production settings (workers=2, pin_memory=true, grad_accum=4).",
        "benchmark": {},
    },
    {
        "name": "dl_workers8_persistent",
        "description": "DataLoader: num_workers=8, persistent_workers, prefetch_factor=4.",
        "benchmark": {
            "dataloader_num_workers": 8,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
            "dataloader_pin_memory": True,
        },
    },
    {
        "name": "dl8_zip_collate_cache",
        "description": "DataLoader x8 + per-worker feature cache + cached collate processor.",
        "benchmark": {
            "dataloader_num_workers": 8,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
            "dataloader_pin_memory": True,
            "optimizations": {"zip_cache": True, "collate_cache": True},
        },
    },
    {
        "name": "dl8_zip_collate_tf32",
        "description": "Above + TF32 + cudnn.benchmark + matmul high precision.",
        "benchmark": {
            "dataloader_num_workers": 8,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
            "dataloader_pin_memory": True,
            "optimizations": {
                "zip_cache": True,
                "collate_cache": True,
                "tf32": True,
                "cudnn_benchmark": True,
                "matmul_precision": "high",
            },
        },
    },
    {
        "name": "dl8_no_cam_text",
        "description": "Disable include_camera_ids/frame_nums to shorten preprocess.",
        "benchmark": {
            "dataloader_num_workers": 8,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
            "dataloader_pin_memory": True,
            "optimizations": {"zip_cache": True, "collate_cache": True, "tf32": True, "cudnn_benchmark": True},
        },
        "data": {
            "train_dataset": {
                "vla_preprocess_args": {
                    "include_camera_ids": False,
                    "include_frame_nums": False,
                }
            }
        },
    },
    {
        "name": "dl8_grad_accum1",
        "description": "Best data path + gradient_accumulation_steps=1 (more optimizer steps, less stall).",
        "benchmark": {
            "dataloader_num_workers": 8,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
            "dataloader_pin_memory": True,
            "gradient_accumulation_steps": 1,
            "optimizations": {"zip_cache": True, "collate_cache": True, "tf32": True, "cudnn_benchmark": True},
        },
        "data": {
            "train_dataset": {
                "vla_preprocess_args": {
                    "include_camera_ids": False,
                    "include_frame_nums": False,
                }
            }
        },
    },
    {
        "name": "dl8_no_grad_ckpt",
        "description": "Disable gradient checkpointing for faster forward (higher VRAM).",
        "benchmark": {
            "dataloader_num_workers": 8,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
            "dataloader_pin_memory": True,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "optimizations": {"zip_cache": True, "collate_cache": True, "tf32": True, "cudnn_benchmark": True},
        },
        "data": {
            "train_dataset": {
                "vla_preprocess_args": {
                    "include_camera_ids": False,
                    "include_frame_nums": False,
                }
            }
        },
    },
    {
        "name": "dl8_torch_compile",
        "description": "torch.compile(inductor) on top of best data settings.",
        "benchmark": {
            "dataloader_num_workers": 8,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
            "dataloader_pin_memory": True,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "torch_compile": True,
            "torch_compile_backend": "inductor",
            "torch_compile_mode": "reduce-overhead",
            "optimizations": {"zip_cache": True, "collate_cache": True, "tf32": True, "cudnn_benchmark": True},
        },
        "data": {
            "train_dataset": {
                "vla_preprocess_args": {
                    "include_camera_ids": False,
                    "include_frame_nums": False,
                }
            }
        },
    },
    {
        "name": "dl8_channels_last",
        "description": "channels_last on visual encoder + best stack so far.",
        "benchmark": {
            "dataloader_num_workers": 8,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
            "dataloader_pin_memory": True,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "optimizations": {
                "zip_cache": True,
                "collate_cache": True,
                "tf32": True,
                "cudnn_benchmark": True,
                "channels_last": True,
            },
        },
        "data": {
            "train_dataset": {
                "vla_preprocess_args": {
                    "include_camera_ids": False,
                    "include_frame_nums": False,
                }
            }
        },
    },
    {
        "name": "best_preload_ga4",
        "description": "Preload cache + dl8 stack, keep grad_accum=4 (same step semantics as baseline).",
        "benchmark": {
            "dataloader_num_workers": 8,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
            "dataloader_pin_memory": True,
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": False,
            "optimizations": {
                "zip_cache": True,
                "collate_cache": True,
                "tf32": True,
                "cudnn_benchmark": True,
                "preload_dataset_cache": True,
            },
        },
        "data": {
            "train_dataset": {
                "vla_preprocess_args": {
                    "include_camera_ids": False,
                    "include_frame_nums": False,
                }
            }
        },
    },
    {
        "name": "best_preload_cache",
        "description": "Best stack + preload all 20 nav samples into RAM (eliminates repeat I/O).",
        "benchmark": {
            "dataloader_num_workers": 4,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 2,
            "dataloader_pin_memory": True,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "optimizations": {
                "zip_cache": True,
                "collate_cache": True,
                "tf32": True,
                "cudnn_benchmark": True,
                "preload_dataset_cache": True,
            },
        },
        "data": {
            "train_dataset": {
                "vla_preprocess_args": {
                    "include_camera_ids": False,
                    "include_frame_nums": False,
                }
            }
        },
    },
    {
        "name": "best_no_deepspeed",
        "description": "Best stack without DeepSpeed ZeRO-2 (native DDP, less comm overhead).",
        "benchmark": {
            "dataloader_num_workers": 4,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 2,
            "dataloader_pin_memory": True,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "optimizations": {
                "zip_cache": True,
                "collate_cache": True,
                "tf32": True,
                "cudnn_benchmark": True,
                "preload_dataset_cache": True,
                "no_deepspeed": True,
            },
        },
        "data": {
            "train_dataset": {
                "vla_preprocess_args": {
                    "include_camera_ids": False,
                    "include_frame_nums": False,
                }
            }
        },
    },
    {
        "name": "production_opt",
        "description": "Recommended production stack (dl8 + zip/collate cache + tf32).",
        "benchmark": {
            "dataloader_num_workers": 8,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
            "dataloader_pin_memory": True,
            "optimizations": {
                "zip_cache": True,
                "collate_cache": True,
                "tf32": True,
                "cudnn_benchmark": True,
            },
        },
    },
]


def run_one(experiment: dict, baseline_sec: float | None) -> dict:
    name = experiment["name"]
    log_path = RESULTS_DIR / f"{name}.log"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    overrides = hydra_overrides(experiment)
    cmd = f"""
source {VENV} && cd {ROOT} && \
torchrun --nproc_per_node {NPROC} -m alpamayo1_5_sft.bench_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_bench \
  {' '.join(overrides)}
"""
    print(f"\n=== Running {name} ===", flush=True)
    t0 = time.time()
    proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    wall = time.time() - t0
    log_path.write_text(proc.stdout + "\n" + proc.stderr)
    result_path = RESULTS_DIR / f"{name}.json"
    if proc.returncode != 0 or not result_path.exists():
        return {
            "name": name,
            "status": "failed",
            "returncode": proc.returncode,
            "wall_sec": wall,
            "log_path": str(log_path),
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
    with result_path.open() as f:
        result = json.load(f)
    avg = float(result["avg_step_sec_stable_tail"])
    gain = (baseline_sec / avg - 1.0) if baseline_sec else 0.0
    record = {
        "name": name,
        "description": experiment.get("description", ""),
        "status": "ok",
        "avg_step_sec_stable_tail": avg,
        "throughput_gain_vs_baseline": gain,
        "wall_sec": wall,
        "result_path": str(result_path),
        "log_path": str(log_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": experiment,
    }
    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def _load_meta(p: Path) -> dict:
    """Pull the comparable knobs out of a result json so the table can show them."""
    with p.open() as f:
        data = json.load(f)
    overrides = data.get("config", {}).get("trainer_overrides", {}) or {}
    opt = data.get("config", {}).get("optimizations", {}) or {}
    return {
        "name": data.get("run_name", p.stem),
        "avg": float(data["avg_step_sec_stable_tail"]),
        "grad_accum": int(data.get("gradient_accumulation_steps", overrides.get("gradient_accumulation_steps", 1))),
        "grad_ckpt": bool(overrides.get("gradient_checkpointing", True)),
        "workers": int(data.get("num_workers", overrides.get("dataloader_num_workers", 0))),
        "optimizations": opt,
    }


def _fmt_opt(opt: dict) -> str:
    keys = [
        ("zip_cache", "zip"),
        ("collate_cache", "collate"),
        ("tf32", "tf32"),
        ("cudnn_benchmark", "cudnn"),
        ("channels_last", "ch_last"),
        ("preload_dataset_cache", "preload"),
        ("no_deepspeed", "no_ds"),
    ]
    enabled = [label for k, label in keys if opt.get(k)]
    mp = opt.get("matmul_precision")
    if mp:
        enabled.append(f"matmul={mp}")
    return ",".join(enabled) if enabled else "-"


def write_markdown(rows: list[dict], baseline_sec: float):
    target_sec = baseline_sec / (1.0 + TARGET_THROUGHPUT_GAIN)
    merged: dict[str, dict] = {}
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            merged[p.stem] = _load_meta(p)
        except Exception:
            continue
    failed = [r for r in rows if r.get("status") != "ok"]
    baseline_ga = merged.get("baseline", {}).get("grad_accum", 4)
    ga4 = sorted([m for m in merged.values() if m["grad_accum"] == baseline_ga], key=lambda m: m["avg"])
    other = sorted([m for m in merged.values() if m["grad_accum"] != baseline_ga], key=lambda m: m["avg"])

    def _row(m: dict, comparable: bool) -> str:
        gain = baseline_sec / m["avg"] - 1.0 if comparable else None
        gain_str = f"{gain*100:+.1f}%" if gain is not None else "n/a*"
        ckpt = "on" if m["grad_ckpt"] else "off"
        return (
            f"| {m['name']} | {m['avg']:.3f} | {gain_str} | {m['grad_accum']} | "
            f"{ckpt} | {m['workers']} | {_fmt_opt(m['optimizations'])} |"
        )

    lines = [
        "# AR1.5 SFT Throughput Benchmark",
        "",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        f"Hardware: {NPROC}x GPU, single node",
        f"Method: 15 steps/run, average of last 10 stable steps",
        f"Baseline avg step: **{baseline_sec:.3f}s** (gradient_accumulation_steps={baseline_ga})",
        f"Target (+80% throughput): **{target_sec:.3f}s** avg step",
        "",
        "> ⚠️ `vs Baseline` is only meaningful for runs with the same `grad_accum` as baseline.",
        "> Rows with a different `grad_accum` show `n/a*` and are listed in section B for reference.",
        "",
        f"## A. Apples-to-apples (grad_accum={baseline_ga}, comparable to baseline)",
        "",
        "| Run | Avg Step (s) | vs Baseline | grad_accum | grad_ckpt | workers | Optimizations |",
        "|-----|-------------:|------------:|-----------:|:---------:|--------:|---------------|",
    ]
    for m in ga4:
        lines.append(_row(m, comparable=True))
    if other:
        lines += [
            "",
            "## B. Different batch semantics (NOT directly comparable to baseline)",
            "",
            "Each `step` only covers 1 micro-batch instead of "
            f"{baseline_ga}; raw step time looks lower but the per-optimizer-step work is smaller.",
            "",
            "| Run | Avg Step (s) | vs Baseline | grad_accum | grad_ckpt | workers | Optimizations |",
            "|-----|-------------:|------------:|-----------:|:---------:|--------:|---------------|",
        ]
        for m in other:
            lines.append(_row(m, comparable=False))

    if failed:
        lines += ["", "## C. Failed runs (this session)", "", "| Run | Log |", "|-----|-----|"]
        for r in failed:
            lines.append(f"| {r['name']} | {r.get('log_path','')} |")

    # Static caveats discovered during the benchmark sweep.
    lines += [
        "",
        "## D. Known incompatibilities / caveats",
        "",
        "| Item | Detail |",
        "|------|--------|",
        "| `dl8_torch_compile` | `torch.compile(inductor)` fails on Qwen3-VL because Dynamo "
        "cannot fake-trace `flash_attn._flash_attn_varlen_forward`. Re-enable after "
        "transformers/flash-attn add Dynamo support. See `results/dl8_torch_compile.log`. |",
        "| `channels_last` on Qwen3-VL visual encoder | Qwen3-VL ViT uses Linear patch embedding "
        "(no Conv2d). `apply_model_optimizations` reports `channels_last_visual_conv2d=0`, "
        "so the flag is effectively a **no-op** on this model. The speedup observed in "
        "`dl8_channels_last` comes from its other knobs (`grad_accum=1` + `grad_ckpt=off`), "
        "not from channels_last. |",
        "| `preload_dataset_cache` | Loads the **entire** train dataset into per-worker RAM. "
        "Safe only for this 20-sample nav demo; do NOT enable in production. |",
        "| `dl8_no_cam_text` regression vs `dl8_zip_collate_tf32` | The former omits "
        "`matmul_precision=high` (intentional, see `EXPERIMENTS`), which is why disabling "
        "`include_camera_ids/frame_nums` does NOT translate into a speedup here. |",
    ]

    best = min(ga4, key=lambda m: m["avg"], default=None)
    if best:
        lines += [
            "",
            f"**Best comparable (grad_accum={baseline_ga}):** `{best['name']}` @ {best['avg']:.3f}s "
            f"({(baseline_sec/best['avg']-1)*100:+.1f}% throughput vs baseline).",
            "",
            "**Production-safe recommendation:** `production_opt` (no train-semantics change, "
            "no out-of-RAM risk, no grad-ckpt trade-off).",
        ]
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_baseline_sec() -> float | None:
    path = RESULTS_DIR / "baseline.json"
    if path.exists():
        with path.open() as f:
            return float(json.load(f)["avg_step_sec_stable_tail"])
    return None


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    exps = [e for e in EXPERIMENTS if (only is None or e["name"] in only)]
    rows: list[dict] = []
    baseline_sec: float | None = load_baseline_sec()
    if baseline_sec is None and (only is None or "baseline" in only):
        pass  # will set after baseline run
    elif baseline_sec is None:
        print("ERROR: benchmark/results/baseline.json missing. Run baseline first.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Using existing baseline: {baseline_sec:.3f}s/step", flush=True)

    for exp in exps:
        row = run_one(exp, baseline_sec)
        rows.append(row)
        if row.get("status") == "ok" and exp["name"] == "baseline":
            baseline_sec = row["avg_step_sec_stable_tail"]
        if baseline_sec and row.get("status") == "ok":
            row["throughput_gain_vs_baseline"] = baseline_sec / row["avg_step_sec_stable_tail"] - 1.0
        if baseline_sec:
            write_markdown(rows, baseline_sec)
            ok_rows = [r for r in rows if r.get("status") == "ok"]
            if ok_rows:
                best = min(ok_rows, key=lambda x: x["avg_step_sec_stable_tail"])
                if (
                    not FULL_SUITE
                    and best["avg_step_sec_stable_tail"] <= baseline_sec / (1.0 + TARGET_THROUGHPUT_GAIN)
                ):
                    print(
                        f"\n*** TARGET REACHED: {best['name']} "
                        f"{best['avg_step_sec_stable_tail']:.3f}s "
                        f"(+{(baseline_sec/best['avg_step_sec_stable_tail']-1)*100:.1f}% throughput) ***",
                        flush=True,
                    )
                    break
    if baseline_sec:
        write_markdown(rows, baseline_sec)
    elif any(r.get("name") == "baseline" and r.get("status") == "ok" for r in rows):
        baseline_sec = next(r["avg_step_sec_stable_tail"] for r in rows if r["name"] == "baseline")
        write_markdown(rows, baseline_sec)
    else:
        print("Baseline failed; see logs in benchmark/results/", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
