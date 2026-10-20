# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch one fresh Hydra/torchrun benchmark and verify its completion.

Run variants as separate invocations, passing their model/data/performance inputs
as repeated --override arguments. No machine, checkpoint, environment, or dataset
paths are assumed. Profile runs get their own result directory and mode label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analyze import summarize


CALLBACK_TARGET = "alpamayo1_5_sft.benchmark_h20.callback.DistributedStepCallback"


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_snapshot(code_root: Path, run_dir: Path) -> dict[str, Any]:
    def git(*arguments: str) -> bytes:
        return subprocess.run(["git", "-C", str(code_root), *arguments], capture_output=True, check=True).stdout

    head = git("rev-parse", "HEAD").decode().strip()
    status = git("status", "--porcelain=v1").decode()
    diff = git("diff", "--no-ext-diff", "--binary", "HEAD")
    (run_dir / "code.patch").write_bytes(diff)
    # HEAD plus tracked diff alone omits an untracked new callback. Preserve this
    # harness verbatim and hash each source, independent of its Git tracking state.
    snapshot = run_dir / "harness_source"
    snapshot.mkdir()
    harness_sha256 = {}
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.is_file() and path.suffix in {".py", ".md"}:
            data = path.read_bytes()
            (snapshot / path.name).write_bytes(data)
            harness_sha256[path.name] = hashlib.sha256(data).hexdigest()
    return {
        "root": str(code_root), "head": head, "dirty": bool(status), "status": status,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "harness_sha256": harness_sha256,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Training environment's Python executable")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, help="Git checkout to snapshot; defaults to work-dir")
    parser.add_argument("--train-module", default="alpamayo1_5_sft.train_hf")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--result-dir", type=Path, required=True, help="Parent for a new, never reused run directory")
    parser.add_argument("--run-id", help="Unique simple directory name; generated if omitted")
    parser.add_argument("--mode", choices=("timing", "profile"), default="timing")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--master-port", type=int, help="Omit for torchrun --standalone with an automatic port")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--steady-start-step", type=int, default=5)
    parser.add_argument("--steady-end-step", type=int, default=20)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--profile-start-step", type=int, default=5)
    parser.add_argument("--profile-end-step", type=int, default=8)
    parser.add_argument("--global-samples-per-step", type=float)
    parser.add_argument("--global-tokens-per-step", type=float)
    parser.add_argument("--override", action="append", default=[], help="One exact Hydra override, repeatable")
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE", help="Child environment override, repeatable; values are not snapshotted")
    parser.add_argument("--nsys", default="nsys", help="Nsight Systems executable used only in profile mode")
    parser.add_argument("--nsys-option", action="append", default=[], help="Extra nsys profile option, e.g. --nsys-option=--sample=none")
    args = parser.parse_args(argv)
    if not 2 <= args.steady_start_step <= args.steady_end_step <= args.max_steps:
        parser.error("Require 2 <= steady-start-step <= steady-end-step <= max-steps")
    if args.mode == "profile" and not 1 <= args.profile_start_step <= args.profile_end_step <= args.max_steps:
        parser.error("Profile step window must fit inside max-steps")
    if args.nproc_per_node < 1 or args.logging_steps < 1:
        parser.error("nproc-per-node and logging-steps must be positive")
    protected = {"trainer.max_steps", "trainer.logging_steps", "paths.output_dir", "trainer.output_dir"}
    for override in args.override:
        key = override.split("=", 1)[0].lstrip("+~")
        if key in protected or key.startswith("callbacks.h20_benchmark"):
            parser.error(f"Use the runner CLI instead of overriding protected key {key}")
    for item in args.env:
        if "=" not in item or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item.split("=", 1)[0]):
            parser.error("--env must be NAME=VALUE with a valid environment variable name")
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("run-id must be a simple directory name")
    return args


def hydra_overrides(args: argparse.Namespace, run_dir: Path, run_id: str) -> list[str]:
    callback = {
        "_target_": CALLBACK_TARGET, "output_path": str(run_dir / "steps.json"), "run_id": run_id,
        "steady_start_step": args.steady_start_step, "steady_end_step": args.steady_end_step,
        "expected_max_steps": args.max_steps, "profile": args.mode == "profile",
        "profile_start_step": args.profile_start_step, "profile_end_step": args.profile_end_step,
        "global_samples_per_step": args.global_samples_per_step,
        "global_tokens_per_step": args.global_tokens_per_step,
    }
    values = [
        *args.override,
        f"++trainer.max_steps={args.max_steps}", f"++trainer.logging_steps={args.logging_steps}",
        "++paths.output_dir=" + json.dumps(str(run_dir / "training")),
        "++trainer.output_dir=" + json.dumps(str(run_dir / "training")),
    ]
    values.extend(f"++callbacks.h20_benchmark.{key}={json.dumps(value)}" for key, value in callback.items())
    return values


def verify_result(result: dict[str, Any], args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    if result.get("run_id") != run_id:
        raise RuntimeError("Result run_id does not match this launch")
    if result.get("mode") != args.mode or result.get("world_size") != args.nproc_per_node:
        raise RuntimeError("Result mode/world size does not match this launch")
    if result.get("expected_max_steps") != args.max_steps:
        raise RuntimeError("Result expected_max_steps does not match this launch")
    if (result.get("steady_start_step"), result.get("steady_end_step")) != (args.steady_start_step, args.steady_end_step):
        raise RuntimeError("Result measurement window does not match this launch")
    if not result.get("complete"):
        raise RuntimeError("Training exited without completing the requested steps/capture")
    for rank in result.get("ranks", []):
        if rank.get("global_step") != args.max_steps:
            raise RuntimeError("A rank did not end at exactly the requested max_steps")
        if args.mode == "profile" and (
            rank.get("capture_started_at_step") != args.profile_start_step
            or rank.get("capture_stopped_at_step") != args.profile_end_step
        ):
            raise RuntimeError("A rank did not capture the exact requested optimizer steps")
    return summarize(result)


def run(args: argparse.Namespace) -> Path:
    work_dir = args.work_dir.resolve(strict=True)
    code_root = (args.code_root or work_dir).resolve(strict=True)
    python = shutil.which(args.python)
    if python is None:
        raise FileNotFoundError(f"Python executable not found: {args.python}")
    if args.mode == "profile" and shutil.which(args.nsys) is None:
        raise FileNotFoundError(f"Nsight Systems executable not found: {args.nsys}")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    run_dir = args.result_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # Never accept a stale JSON from an earlier attempt.
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "run_id": run_id, "mode": args.mode, "status": "preparing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "work_dir": str(work_dir), "python": python,
    }
    write_json(manifest_path, manifest)
    try:
        manifest["code"] = git_snapshot(code_root, run_dir)
        config_flags = ["--config-path", args.config_path, "--config-name", args.config_name]
        overrides = hydra_overrides(args, run_dir, run_id)
        env = os.environ.copy()
        env.update(dict(item.split("=", 1) for item in args.env))
        # Editable namespace-package finders can override PYTHONPATH and silently
        # benchmark a different checkout. Verify the actual loader origins first.
        modules = [args.train_module, "alpamayo.data.pai_nav",
                   "alpamayo.processor.qwen_processor",
                   "alpamayo_r1.load_physical_aiavdataset"]
        probe = (
            "import importlib.util,json; "
            "names=" + repr(modules) + "; "
            "print(json.dumps({n:importlib.util.find_spec(n).origin for n in names}))"
        )
        source_probe = subprocess.run([python, "-c", probe], cwd=work_dir,
                                      env=env, capture_output=True, text=True, check=True)
        origins = json.loads(source_probe.stdout)
        for module in modules[:3]:
            if not Path(origins[module]).resolve().is_relative_to(code_root):
                raise RuntimeError(f"Wrong checkout imported for {module}: {origins[module]}")
        manifest["runtime_source_origins"] = origins
        benchmark_keys = (
            "APPLY_VLM_SKIP_HEAD", "APPLY_LIGER_KERNEL", "APPLY_LIGER_SWIGLU",
            "APPLY_COMPILE_EXPERT", "APPLY_PRELOAD", "APPLY_SAMPLE_CACHE",
            "APPLY_DROP_RAW", "APPLY_VLM_KV_CACHE", "APPLY_REPEAT_DATA",
            "APPLY_PRELOAD_MAX_BYTES", "APPLY_NVTX", "OMP_NUM_THREADS",
            "APPLY_VLM_PREFIX_ONLY", "APPLY_TRAJECTORY_FACTOR_CACHE",
            "CUDA_VISIBLE_DEVICES", "TOKENIZERS_PARALLELISM",
        )
        manifest["benchmark_environment"] = {key: env[key] for key in benchmark_keys if key in env}
        # Values may contain credentials; record names only. The resolved Hydra
        # configuration is still a full snapshot of the actual training config.
        manifest["environment_override_names"] = [item.split("=", 1)[0] for item in args.env]
        compose_command = [python, "-m", args.train_module, *config_flags, "--cfg", "job", "--resolve", *overrides]
        manifest["compose_command"] = compose_command
        with (run_dir / "config.resolved.yaml").open("w", encoding="utf-8") as output, (run_dir / "config.stderr.log").open("w", encoding="utf-8") as error:
            composed = subprocess.run(compose_command, cwd=work_dir, env=env, stdout=output, stderr=error, check=False)
        if composed.returncode != 0:
            raise RuntimeError(f"Hydra configuration snapshot failed ({composed.returncode}); inspect config.stderr.log")
        manifest["config_sha256"] = hashlib.sha256((run_dir / "config.resolved.yaml").read_bytes()).hexdigest()
        command = [python, "-m", "torch.distributed.run", "--nproc-per-node", str(args.nproc_per_node)]
        if args.master_port is None:
            command.append("--standalone")
        else:
            command.extend(["--master-port", str(args.master_port)])
        command.extend(["--module", args.train_module, *config_flags, *overrides])
        if args.mode == "profile":
            command = [
                args.nsys, "profile", "--trace=cuda,nvtx,osrt,cublas,cudnn",
                "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                "--force-overwrite=false", "--output=" + str(run_dir / "nsight"),
                *args.nsys_option, *command,
            ]
        manifest.update({"command": command, "status": "running"})
        write_json(manifest_path, manifest)
        print(f"Run directory: {run_dir}", flush=True)
        print(shlex.join(command), flush=True)
        with (run_dir / "train.log").open("w", encoding="utf-8") as output:
            completed = subprocess.run(command, cwd=work_dir, env=env, stdout=output, stderr=subprocess.STDOUT, check=False)
        manifest["returncode"] = completed.returncode
        if completed.returncode != 0:
            raise RuntimeError(f"Training failed with exit code {completed.returncode}; inspect train.log")
        result = json.loads((run_dir / "steps.json").read_text(encoding="utf-8"))
        summary = verify_result(result, args, run_id)
        if args.mode == "profile":
            reports = sorted(run_dir.glob("*.nsys-rep"))
            if not reports:
                raise RuntimeError("Profile command exited without producing a .nsys-rep artifact")
            manifest["nsight_reports"] = [str(path) for path in reports]
        write_json(run_dir / "summary.json", summary)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
    print(json.dumps(summary["rank_max"], indent=2), flush=True)
    return run_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
