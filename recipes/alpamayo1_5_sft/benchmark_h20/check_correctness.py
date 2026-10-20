# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare real Stage-2 loss/gradients on one checkpoint and one PAI nav sample.

Example, from the checkout with the recipe environment active::

    PYTHONPATH=recipes:src python -m alpamayo1_5_sft.benchmark_h20.check_correctness \
      --checkpoint /path/to/checkpoint --data /path/to/pai \
      --annotations /path/to/nav.json --chunk-ids '[214,224]' \
      --output /path/to/new-correctness.json

The same loaded weights and collated sample are used for every variant, in the
order baseline -> skiphead -> prefix -> trajectory -> Liger, omitting optional
variants unless requested. The default remains baseline, skiphead and Liger.
Each named candidate is compared independently against baseline with skip-head
enabled; prefix and trajectory switches are disabled again for later candidates.
Liger runs after the isolated candidates because its patch cannot be undone.
An optional final ``combined`` variant enables skip-head, prefix, factor reuse
and Liger together. Trajectory variants warm the factors, then check cache hits
with loss and gradients against baseline after resetting the RNGs.
No optimizer is created or stepped.
Python, NumPy and CPU/CUDA torch RNGs are reset before each forward/backward.
Baseline gradients are retained in CPU float32 (roughly 9 GB for a 2.3B expert).
The comparison is numerical under CUDA bf16 autocast, with explicit configurable
tolerances; it does not assert bitwise equivalence or convergence equivalence.
All heavy dependencies are imported after CLI parsing, so --help works alone.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SWITCHES = (
    "APPLY_VLM_SKIP_HEAD", "APPLY_LIGER_KERNEL", "APPLY_LIGER_SWIGLU",
    "APPLY_COMPILE_EXPERT", "APPLY_VLM_KV_CACHE", "APPLY_PROFILE", "APPLY_NVTX",
    "APPLY_PRELOAD", "APPLY_SAMPLE_CACHE", "APPLY_DROP_RAW", "APPLY_REPEAT_DATA",
    "APPLY_PRELOAD_MAX_BYTES", "APPLY_VLM_PREFIX_ONLY", "APPLY_TRAJECTORY_FACTOR_CACHE",
)


def variant_switches(name: str, liger_swiglu: bool) -> dict[str, str]:
    """Describe isolated candidate composition without importing model dependencies."""
    if name not in {"baseline", "skiphead", "prefix", "trajectory", "liger", "combined"}:
        raise ValueError(f"Unknown correctness variant: {name}")
    values = {switch: "0" for switch in SWITCHES}
    values["APPLY_REPEAT_DATA"] = "1"
    values["APPLY_PRELOAD_MAX_BYTES"] = str(4 * 1024**3)
    values["APPLY_VLM_SKIP_HEAD"] = "0" if name == "baseline" else "1"
    values["APPLY_VLM_PREFIX_ONLY"] = str(int(name in {"prefix", "combined"}))
    values["APPLY_TRAJECTORY_FACTOR_CACHE"] = str(int(name in {"trajectory", "combined"}))
    values["APPLY_LIGER_KERNEL"] = str(int(name in {"liger", "combined"}))
    values["APPLY_LIGER_SWIGLU"] = str(int(liger_swiglu))
    return values


def trajectory_operations(stats: dict[str, Any] | None) -> int:
    """Count factor lookups, so a no-op candidate cannot pass as an optimization."""
    return sum(
        sum(instance.get(kind, {}).values())
        for instance in (stats or {}).get("instances", [])
        for kind in ("hits", "misses")
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Existing local Alpamayo checkpoint directory")
    parser.add_argument("--data", "--pai-dir", dest="data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--chunk-ids", required=True, help="JSON integer list, comma-separated IDs, or PAI half-open range such as 0-9")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an earlier correctness report")
    parser.add_argument("--repo-dir", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--sample-index", type=int, default=0, help="Index after nav chunk filtering")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--variants", choices=("skiphead", "prefix", "trajectory", "liger", "combined"), nargs="+",
        default=["skiphead", "liger"],
        help="Baseline comparisons, ordered skiphead/prefix/trajectory/Liger/combined",
    )
    parser.add_argument("--liger-swiglu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None, help="Omit to use the recipe performance setting")
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--loss-atol", type=float, default=1e-3)
    parser.add_argument("--loss-rtol", type=float, default=2e-2)
    parser.add_argument("--grad-rel-l2-tol", type=float, default=5e-2, help="Maximum global ||g-g_ref||2 / ||g_ref||2")
    parser.add_argument("--grad-max-abs-tol", type=float, default=5e-2, help="Maximum absolute difference across every gradient element")
    parser.add_argument("--allow-missing-gradients", action="store_true", help="Allow parameters unused in the baseline only when each variant has the same missing pattern")
    parser.add_argument("--override", action="append", default=[], help="Additional Hydra config override, repeatable; resolved model/data configs are recorded")
    args = parser.parse_args(argv)
    if args.sample_index < 0:
        parser.error("sample-index must be nonnegative")
    for name in ("loss_atol", "loss_rtol", "grad_rel_l2_tol", "grad_max_abs_tol"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name.replace('_', '-')} must be finite and nonnegative")
    return args


def chunk_ids(value: str) -> list[int] | str:
    if re.fullmatch(r"\d+-\d+", value):
        start, end = map(int, value.split("-"))
        if end <= start:
            raise ValueError("PAI chunk range must have end > start; its upper bound is exclusive")
        return value
    parsed = json.loads(value) if value.lstrip().startswith("[") else [int(part) for part in value.split(",")]
    if not isinstance(parsed, list) or not parsed or any(type(item) is not int or item < 0 for item in parsed):
        raise ValueError("chunk-ids must be a nonempty list of nonnegative integers")
    return parsed


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def finite_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def seed_all(seed: int, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clone_to(value: Any, device: Any, torch: Any) -> Any:
    """Clone every tensor: model.forward pops input_ids from its input mapping."""
    if torch.is_tensor(value):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return {key: clone_to(item, device, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to(item, device, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to(item, device, torch) for item in value)
    return value


def input_description(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, dict):
        return {str(key): input_description(item, torch) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # Retain every tensor shape, but avoid dumping prompt or annotation text.
        return {"length": len(value), "items": [input_description(item, torch) for item in value]}
    return {"type": type(value).__name__}


def gradient_metrics(current: Any, reference: Any | None, torch: Any) -> dict[str, Any]:
    """Accumulate double precision statistics in bounded CPU chunks."""
    flat = current.reshape(-1)
    ref = reference.reshape(-1) if reference is not None else None
    if ref is not None and current.shape != reference.shape:
        raise ValueError("Gradient shape changed between variants")
    norm2 = ref_norm2 = diff_norm2 = 0.0
    max_abs = 0.0
    finite = True
    for start in range(0, flat.numel(), 1_048_576):
        values = flat[start:start + 1_048_576].double()
        if not bool(torch.isfinite(values).all()):
            finite = False
            continue
        norm2 += float(values.square().sum())
        if ref is not None:
            ref_values = ref[start:start + 1_048_576].double()
            if not bool(torch.isfinite(ref_values).all()):
                finite = False
                continue
            difference = values - ref_values
            ref_norm2 += float(ref_values.square().sum())
            diff_norm2 += float(difference.square().sum())
            if difference.numel():
                max_abs = max(max_abs, float(difference.abs().max()))
    relative_l2 = math.sqrt(diff_norm2 / ref_norm2) if ref_norm2 else (0.0 if diff_norm2 == 0 else None)
    return {
        "finite": finite,
        "norm_l2": finite_number(math.sqrt(norm2)) if finite else None,
        "reference_norm_l2": math.sqrt(ref_norm2) if finite and ref is not None else None,
        "difference_norm_l2": math.sqrt(diff_norm2) if finite and ref is not None else None,
        "relative_l2_error": relative_l2 if finite and ref is not None else None,
        "max_abs_error": max_abs if finite and ref is not None else None,
        "_norm2": norm2,
        "_ref_norm2": ref_norm2,
        "_diff_norm2": diff_norm2,
    }


def collect_gradients(
    model: Any, reference: dict[str, Any | None] | None, torch: Any,
) -> tuple[dict[str, Any], dict[str, Any | None] | None]:
    """Save all baseline gradients; process later variants one parameter at a time."""
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("Model has no trainable parameters")
    if reference is not None and set(reference) != {name for name, _ in parameters}:
        raise RuntimeError("The trainable parameter set changed between variants")
    backup: dict[str, Any | None] | None = {} if reference is None else None
    rows = []
    missing = []
    pattern_matches = True
    total_norm2 = total_ref_norm2 = total_diff_norm2 = 0.0
    global_max_abs = 0.0
    all_finite = True
    backup_bytes = 0
    for name, parameter in parameters:
        grad = parameter.grad
        row: dict[str, Any] = {
            "name": name, "shape": list(parameter.shape), "numel": parameter.numel(),
            "parameter_dtype": str(parameter.dtype), "gradient_present": grad is not None,
            "gradient_dtype": str(grad.dtype) if grad is not None else None,
        }
        reference_grad = reference[name] if reference is not None else None
        if grad is None:
            missing.append(name)
            row["finite"] = None
            if backup is not None:
                backup[name] = None
            if reference is not None and reference_grad is not None:
                pattern_matches = False
        else:
            if grad.is_sparse:
                raise TypeError(f"Dense Stage-2 gradient expected for {name}")
            cpu_grad = grad.detach().to(device="cpu", dtype=torch.float32, copy=True)
            stats = gradient_metrics(cpu_grad, reference_grad, torch)
            total_norm2 += stats.pop("_norm2")
            total_ref_norm2 += stats.pop("_ref_norm2")
            total_diff_norm2 += stats.pop("_diff_norm2")
            all_finite = all_finite and stats["finite"]
            if stats["max_abs_error"] is not None:
                global_max_abs = max(global_max_abs, stats["max_abs_error"])
            row.update(stats)
            if backup is not None:
                backup[name] = cpu_grad
                backup_bytes += cpu_grad.numel() * cpu_grad.element_size()
            elif reference_grad is None:
                pattern_matches = False
            del cpu_grad
        rows.append(row)

    relative_l2 = math.sqrt(total_diff_norm2 / total_ref_norm2) if total_ref_norm2 else (0.0 if total_diff_norm2 == 0 else None)
    report = {
        "trainable_parameter_tensors": len(parameters),
        "trainable_parameter_elements": sum(parameter.numel() for _, parameter in parameters),
        "all_present_gradients_finite": all_finite,
        "missing_gradient_names": missing,
        "missing_pattern_matches_baseline": pattern_matches,
        "global_norm_l2": finite_number(math.sqrt(total_norm2)) if all_finite else None,
        "global_reference_norm_l2": math.sqrt(total_ref_norm2) if all_finite and reference is not None else None,
        "global_difference_norm_l2": math.sqrt(total_diff_norm2) if all_finite and reference is not None else None,
        "global_relative_l2_error": relative_l2 if all_finite and reference is not None else None,
        "max_abs_error": global_max_abs if all_finite and reference is not None else None,
        "baseline_gradient_cpu_bytes": backup_bytes if backup is not None else None,
        "parameters": rows,
    }
    return report, backup


def vlm_freeze_report(model: Any) -> dict[str, Any]:
    vlm_parameters = list(model.vlm.named_parameters())
    if not vlm_parameters:
        raise RuntimeError("VLM has no parameters; a real loaded VLM is required")
    unfrozen = [name for name, parameter in vlm_parameters if parameter.requires_grad]
    gradients = [name for name, parameter in vlm_parameters if parameter.grad is not None]
    return {
        "cotrain_vlm": bool(model.cotrain_vlm),
        "parameter_tensors": len(vlm_parameters),
        "parameter_elements": sum(parameter.numel() for _, parameter in vlm_parameters),
        "all_parameters_frozen": not unfrozen,
        "all_gradients_none": not gradients,
        "unfrozen_parameter_names": unfrozen,
        "parameters_with_gradients": gradients,
    }


def run(args: argparse.Namespace, report: dict[str, Any]) -> bool:
    # Recipe/source paths are explicit so direct script invocation works as well
    # as python -m. The checkpoint/dataset/model are never substituted by stubs.
    repo = args.repo_dir.resolve(strict=True)
    sys.path[:0] = [str(repo / "recipes"), str(repo / "src")]
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from alpamayo1_5_sft.performance.apply_optimizations import apply_runtime_optimizations
    from alpamayo1_5_sft.performance.stage2_optimizations import apply_stage2_optimizations

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Correctness checking is single GPU; do not launch with distributed torchrun")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU and the real recipe environment are required")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select one CUDA GPU")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU must support CUDA bf16 autocast")

    checkpoint = args.checkpoint.resolve(strict=True)
    data = args.data.resolve(strict=True)
    annotations = args.annotations.resolve(strict=True)
    overrides = [
        "model.pretrained_model_name_or_path=" + json.dumps(str(checkpoint)),
        "model.stage1_vlm_checkpoint_path=null",
        "model.cotrain_vlm=false",
        "data.train_dataset.local_dir=" + json.dumps(str(data)),
        "data.train_dataset.annotations_path=" + json.dumps(str(annotations)),
        "data.train_dataset.chunk_ids=" + json.dumps(chunk_ids(args.chunk_ids)),
        *args.override,
    ]
    with initialize_config_dir(config_dir=str(repo / "recipes/alpamayo1_5_sft/configs"), version_base=None):
        cfg = compose(config_name="sft_stage2_nav", overrides=overrides)
    if cfg.model.cotrain_vlm or cfg.data.train_dataset.vla_preprocess_args.generation_mode:
        raise ValueError("This check requires frozen-VLM Stage-2 training preprocessing")
    performance = OmegaConf.to_container(cfg.get("performance", {}), resolve=True)
    if args.tf32 is not None:
        performance["tf32"] = args.tf32
    if args.cudnn_benchmark is not None:
        performance["cudnn_benchmark"] = args.cudnn_benchmark
    runtime = apply_runtime_optimizations({"optimizations": performance})
    seed_all(args.seed, torch, np)
    print("Loading real Stage-2 checkpoint...", flush=True)
    model = instantiate(cfg.model, _convert_="partial").to(device)
    model.train()
    frozen = vlm_freeze_report(model)
    if not frozen["all_parameters_frozen"] or frozen["cotrain_vlm"]:
        raise RuntimeError("Loaded model does not have a frozen Stage-2 VLM")

    print("Loading and collating one real PAI nav sample...", flush=True)
    seed_all(args.seed, torch, np)
    dataset = instantiate(cfg.data.train_dataset, _convert_="partial", model_config=model.config)
    if args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} exceeds filtered dataset length {len(dataset)}")
    sample = dataset[args.sample_index]
    if sample is None:
        raise RuntimeError("Selected real sample could not be decoded")
    collate = instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)
    cpu_batch = clone_to(collate([sample]), torch.device("cpu"), torch)
    sample_entry = getattr(dataset, "_samples", [])[args.sample_index]
    report.update({
        "resolved_model_config": OmegaConf.to_container(cfg.model, resolve=True),
        "resolved_train_dataset_config": OmegaConf.to_container(cfg.data.train_dataset, resolve=True),
        "sample": {
            "filtered_dataset_length": len(dataset), "sample_index": args.sample_index,
            "clip_id": sample_entry.get("clip_id"), "t0_relative": sample_entry.get("t0_relative"),
        },
        "input": input_description(cpu_batch, torch),
        "vlm_freeze_before": frozen,
        "runtime": runtime,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_cuda_version": torch.version.cuda,
    })
    versions = {}
    for name in ("torch", "transformers", "liger-kernel", "physical-ai-av", "flash-attn", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report["versions"] = versions
    git = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    report["recipe_commit"] = git.stdout.strip() if git.returncode == 0 else None
    patch = subprocess.run(["git", "-C", str(repo), "diff", "HEAD", "--"], capture_output=True, check=False)
    report["recipe_tracked_patch_sha256"] = hashlib.sha256(patch.stdout).hexdigest() if patch.returncode == 0 else None
    report["source_files"] = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((repo / "recipes/alpamayo1_5_sft/performance").glob("*.py"))
    }
    for relative in ("recipes/alpamayo1_5_sft/models/sft_alpamayo_r1.py", "recipes/alpamayo1_5_sft/benchmark_h20/check_correctness.py"):
        report["source_files"][relative] = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
    del sample, dataset, collate
    gc.collect()
    write_json(args.output, report)

    names = ["baseline"] + [
        name for name in ("skiphead", "prefix", "trajectory", "liger", "combined") if name in args.variants
    ]
    trajectory_apply = trajectory_stats = None
    if any(name in names for name in ("trajectory", "combined")):
        from alpamayo1_5_sft.performance.stage2_trajectory import (
            apply_trajectory_factor_cache,
            trajectory_factor_cache_stats,
        )

        trajectory_apply = apply_trajectory_factor_cache
        trajectory_stats = trajectory_factor_cache_stats
    report["variant_order"] = names
    report["variant_comparison"] = (
        "Each candidate independently compared with baseline on identical weights/input/RNG; "
        "all non-baseline candidates include skip-head; combined enables all candidates after Liger"
    )
    reference_grads = None
    reference_loss = None
    passed = True
    for name in names:
        switches = variant_switches(name, args.liger_swiglu)
        os.environ.update(switches)
        applied = apply_stage2_optimizations(model)
        trajectory_applied = trajectory_apply(model) if trajectory_apply else None
        trajectory_cold_loss = None
        if name in {"trajectory", "combined"}:
            # Validate the actual reuse path, not just the cold factor build.
            # No optimizer is stepped; the formal comparison resets all RNGs.
            seed_all(args.seed, torch, np)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                warmup = model(**clone_to(cpu_batch, device, torch))
            trajectory_cold_loss = float(warmup.loss.detach().float().cpu())
            del warmup
        trajectory_before = trajectory_stats(model) if trajectory_stats else None
        # A stale report must not make a missing prefix integration look active.
        model._stage2_prefix_report = None
        model.zero_grad(set_to_none=True)
        batch = clone_to(cpu_batch, device, torch)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        seed_all(args.seed, torch, np)
        print(f"Checking {name}: real bf16 forward and backward...", flush=True)
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss
        if loss is None or loss.numel() != 1:
            raise RuntimeError(f"{name} did not produce a scalar loss")
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        loss_value = float(loss.detach().float().cpu())
        freeze_after = vlm_freeze_report(model)
        prefix_report = getattr(model, "_stage2_prefix_report", None)
        trajectory_after = trajectory_stats(model) if trajectory_stats else None
        activation_errors = []
        if name in {"prefix", "combined"} and not (prefix_report and prefix_report.get("applied")):
            activation_errors.append("Requested prefix candidate did not trim this real sample")
        if name in {"trajectory", "combined"}:
            if trajectory_operations(trajectory_after) <= trajectory_operations(trajectory_before):
                activation_errors.append("Requested trajectory candidate performed no factor lookups")
            for role in ("theta", "accel"):
                before_hits = sum(i.get("hits", {}).get(role, 0) for i in (trajectory_before or {}).get("instances", []))
                after_hits = sum(i.get("hits", {}).get(role, 0) for i in (trajectory_after or {}).get("instances", []))
                if after_hits <= before_hits:
                    activation_errors.append(f"Requested trajectory candidate did not reuse a cached {role} factor")
            if not any(
                instance.get("entries", 0) > 0
                for instance in (trajectory_after or {}).get("instances", [])
            ):
                activation_errors.append("Requested trajectory candidate cached no factors")
        elif trajectory_before != trajectory_after:
            activation_errors.append("Disabled trajectory cache was used by another candidate")
        gradients, backup = collect_gradients(model, reference_grads, torch)
        if name == "baseline":
            reference_grads = backup
            reference_loss = loss_value
        loss_difference = abs(loss_value - reference_loss)
        loss_limit = args.loss_atol + args.loss_rtol * abs(reference_loss)
        loss_ok = math.isfinite(loss_value) and math.isfinite(reference_loss) and loss_difference <= loss_limit
        if trajectory_cold_loss is not None:
            loss_ok = loss_ok and math.isfinite(trajectory_cold_loss) and abs(trajectory_cold_loss - reference_loss) <= loss_limit
        gradients_ok = gradients["all_present_gradients_finite"] and gradients["missing_pattern_matches_baseline"]
        gradients_ok = gradients_ok and (args.allow_missing_gradients or not gradients["missing_gradient_names"])
        if name != "baseline":
            relative = gradients["global_relative_l2_error"]
            absolute = gradients["max_abs_error"]
            gradients_ok = gradients_ok and relative is not None and relative <= args.grad_rel_l2_tol
            gradients_ok = gradients_ok and absolute is not None and absolute <= args.grad_max_abs_tol
        frozen_ok = freeze_after["all_parameters_frozen"] and freeze_after["all_gradients_none"]
        variant_passed = bool(loss_ok and gradients_ok and frozen_ok and not activation_errors)
        passed = passed and variant_passed
        variant = {
            "name": name, "passed": variant_passed, "applied": applied,
            "switches": switches, "activation_errors": activation_errors,
            "prefix_report": prefix_report,
            "trajectory_factor_cache_applied": trajectory_applied,
            "trajectory_factor_cache_before": trajectory_before,
            "trajectory_factor_cache_after": trajectory_after,
            "trajectory_cold_loss": trajectory_cold_loss,
            "loss": finite_number(loss_value), "loss_finite": math.isfinite(loss_value),
            "loss_abs_difference": finite_number(loss_difference),
            "loss_abs_limit": finite_number(loss_limit), "loss_within_tolerance": loss_ok,
            "gradients_within_tolerance": bool(gradients_ok),
            "vlm_freeze_after": freeze_after, "gradients": gradients,
            "forward_backward_wall_seconds": elapsed,
            "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
        report["variants"].append(variant)
        write_json(args.output, report)
        print(json.dumps({key: variant[key] for key in ("name", "passed", "loss", "loss_abs_difference", "gradients_within_tolerance")}), flush=True)
        del batch, outputs, loss
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    return bool(passed)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output = args.output.resolve()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite an existing report: {args.output}; use --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one real sample, one real checkpoint, no optimizer update; bf16 numerical check",
        "seed": args.seed, "device": args.device,
        "requested_variants": args.variants,
        "tolerances": {
            "loss_atol": args.loss_atol, "loss_rtol": args.loss_rtol,
            "gradient_global_relative_l2": args.grad_rel_l2_tol,
            "gradient_max_absolute_element_error": args.grad_max_abs_tol,
            "allow_missing_baseline_gradients": args.allow_missing_gradients,
            "gradient_acceptance": "finite gradients and matching presence; global relative L2 AND maximum absolute element error must meet limits",
        },
        "variants": [],
    }
    saved_environment = {name: os.environ.get(name) for name in SWITCHES}
    # Explicit isolation prevents inheriting old benchmark flags (especially a
    # default-on skip-head, KV cache, compile or repeated-data setting).
    os.environ.update(variant_switches("baseline", args.liger_swiglu))
    write_json(args.output, report)
    try:
        passed = run(args, report)
        report["status"] = "passed" if passed else "failed"
        report["passed"] = passed
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output, report)
        print(f"Correctness report: {args.output} ({report['status']})", flush=True)
        return 0 if passed else 1
    except Exception as exc:
        report["status"] = "error"
        report["passed"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(args.output, report)
        raise
    finally:
        for name, value in saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
