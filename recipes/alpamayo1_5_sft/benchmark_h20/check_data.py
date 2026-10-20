# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check real PAI preprocessing/collation equivalence without model weights or GPUs.

Compare every filtered navigation annotation using the ordinary uncached dataset
and configured collator against eager CPU preload/drop-raw plus cached collation.
Each source index is read twice without reseeding to detect stochastic outputs;
mutating a returned cached tree must not change its next read. All comparisons
are exact, including tensor shape/dtype/value, prompt text, and token IDs.

Run as ``python -m alpamayo1_5_sft.benchmark_h20.check_data --help``. This command
sets CUDA_VISIBLE_DEVICES='' before importing torch. No model is instantiated,
no checkpoint weight shard is opened, and no GPU experiment is performed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from functools import partial
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Local config.json directory",
    )
    parser.add_argument("--data", type=Path, required=True, help="Local real PAI dataset directory")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--chunk-ids", required=True, help="JSON list, comma IDs, or half-open range",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-dir", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--expected-unique-samples", type=int, default=20)
    parser.add_argument("--preload-max-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--override", action="append", default=[], help="Hydra dataset override")
    args = parser.parse_args(argv)
    for name in ("cpu_threads", "expected_unique_samples", "preload_max_bytes"):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    return args


def write_json(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tensor_digest(value: Any, torch: Any) -> str:
    # Viewing bytes also handles bfloat16, which NumPy cannot represent directly.
    if value.device.type != "cpu":
        raise RuntimeError(f"CPU-only check received a tensor on {value.device}")
    data = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def compare_trees(
    reference: Any, candidate: Any, torch: Any, np: Any, *, manifest: bool = False,
) -> dict[str, Any]:
    """Compare complete trees, allowing only equivalent sequence containers.

    Preload deliberately normalizes Hydra ListConfig metadata to a plain list.
    Compare ordered sequence lengths and recursively exact values rather than
    container classes. Tensor/array branches stay separate and retain their
    exact shape/dtype/value checks; strings and bytes remain exact scalars.
    """
    result: dict[str, Any] = {
        "equal": True, "tensor_leaves": 0, "array_leaves": 0, "scalar_leaves": 0,
        "differences": [],
    }
    if manifest:
        result["tensor_manifest"] = []

    def mismatch(path: str, reason: str) -> None:
        result["equal"] = False
        result["differences"].append({"path": path, "reason": reason})

    def is_sequence(value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))

    def walk(left: Any, right: Any, path: str) -> None:
        if torch.is_tensor(left) or torch.is_tensor(right):
            result["tensor_leaves"] += 1
            if not (torch.is_tensor(left) and torch.is_tensor(right)):
                mismatch(path, "tensor/non-tensor type mismatch")
                return
            if left.device.type != "cpu" or right.device.type != "cpu":
                raise RuntimeError(f"CPU-only check encountered non-CPU tensor at {path}")
            same = left.shape == right.shape and left.dtype == right.dtype
            same = bool(same and torch.equal(left, right))
            if not same:
                mismatch(path, "tensor shape, dtype or exact values differ")
            if manifest:
                result["tensor_manifest"].append({
                    "path": path, "shape": list(left.shape), "dtype": str(left.dtype),
                    "reference_sha256": tensor_digest(left, torch),
                    "candidate_sha256": tensor_digest(right, torch), "equal": same,
                })
        elif isinstance(left, Mapping) or isinstance(right, Mapping):
            if not (isinstance(left, Mapping) and isinstance(right, Mapping)):
                mismatch(path, "mapping/non-mapping type mismatch")
                return
            if set(left) != set(right):
                mismatch(path, "mapping keys differ")
            for key in sorted(set(left) & set(right), key=str):
                walk(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            result["array_leaves"] += 1
            same = isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
            if not (same and left.dtype == right.dtype and np.array_equal(left, right)):
                mismatch(path, "array dtype, shape or exact values differ")
        elif is_sequence(left) or is_sequence(right):
            if not (is_sequence(left) and is_sequence(right)):
                mismatch(path, "sequence/non-sequence type mismatch")
                return
            if len(left) != len(right):
                mismatch(path, "sequence length differs")
                return
            for index, (l_item, r_item) in enumerate(zip(left, right)):
                walk(l_item, r_item, f"{path}[{index}]")
        else:
            result["scalar_leaves"] += 1
            try:
                same = type(left) is type(right) and bool(left == right)
            except (TypeError, ValueError):
                same = False
            if not same:
                mismatch(path, "scalar/text type or value differs")

    walk(reference, candidate, "input")
    return result


def mutate_returned_tree(value: Any, torch: Any, np: Any) -> dict[str, int]:
    """Corrupt one element of every tensor/array, then every mutable container."""
    counts = {"tensors": 0, "arrays": 0, "containers": 0}

    def mutate(item: Any) -> None:
        if torch.is_tensor(item):
            if item.device.type != "cpu":
                raise RuntimeError("Cache returned a non-CPU tensor")
            if item.numel():
                index = (0,) * item.ndim
                item[index] = 1 if item[index].item() == 0 else 0
                counts["tensors"] += 1
        elif isinstance(item, np.ndarray):
            if item.size:
                item.flat[0] = 1 if item.flat[0] == 0 else 0
                counts["arrays"] += 1
        elif isinstance(item, dict):
            for child in list(item.values()):
                mutate(child)
            item.clear()
            counts["containers"] += 1
        elif isinstance(item, list):
            for child in item:
                mutate(child)
            item.clear()
            counts["containers"] += 1
        elif isinstance(item, tuple):
            for child in item:
                mutate(child)

    mutate(value)
    return counts


def model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "tokenized_data", "labels_mask", "ego_history_xyz", "ego_history_rot",
        "ego_future_xyz", "ego_future_rot",
    }
    missing = required - set(batch)
    if missing:
        raise ValueError(f"Collated model inputs missing {sorted(missing)}")
    return {
        key: value for key, value in batch.items()
        if key in required or key.startswith("ego_")
    }


def run(args: argparse.Namespace, report: dict[str, Any]) -> bool:
    repo = args.repo_dir.resolve(strict=True)
    sys.path[:0] = [str(repo / "recipes"), str(repo / "src")]
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from alpamayo_r1.config import AlpamayoR1Config
    from alpamayo.data.pai_nav import PAIDatasetWithNav
    from alpamayo.data.pai_utils import PhysicalAIAVDatasetLocalInterface
    from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
    from alpamayo1_5_sft.performance.collate_cache import collate_fn_from_model_config_cached
    from alpamayo1_5_sft.performance.stage2_data import apply_stage2_training_data
    from alpamayo1_5_sft.benchmark_h20.check_correctness import chunk_ids

    if torch.cuda.is_initialized():
        raise RuntimeError("Refusing a process with an initialized CUDA context")
    if getattr(PhysicalAIAVDatasetLocalInterface, "_zip_cache_enabled", False):
        raise RuntimeError("Reference dataset must use the original uncached feature reader")
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Seed only the CPU generator; CUDA RNG APIs are never called.
    torch.default_generator.manual_seed(args.seed)
    checkpoint = args.checkpoint.resolve(strict=True)
    data = args.data.resolve(strict=True)
    annotations = args.annotations.resolve(strict=True)
    origins = {}
    for name in (
        "alpamayo.data.pai_nav", "alpamayo.processor.qwen_processor",
        "alpamayo1_5_sft.performance.stage2_data",
        "alpamayo1_5_sft.performance.collate_cache",
        "alpamayo1_5_sft.models.sft_alpamayo_r1", "alpamayo_r1.config",
    ):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if not name.startswith("alpamayo_r1.") and not origin.is_relative_to(repo):
            raise RuntimeError(f"Wrong checkout imported for {name}: {origin}")
        origins[name] = {
            "path": str(origin), "sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
        }
    overrides = [
        "data.train_dataset.local_dir=" + json.dumps(str(data)),
        "data.train_dataset.annotations_path=" + json.dumps(str(annotations)),
        "data.train_dataset.chunk_ids=" + json.dumps(chunk_ids(args.chunk_ids)),
        *args.override,
    ]
    with initialize_config_dir(
        config_dir=str(repo / "recipes/alpamayo1_5_sft/configs"), version_base=None,
    ):
        cfg = compose(config_name="sft_stage2_nav", overrides=overrides)
    if cfg.data.train_dataset.vla_preprocess_args.generation_mode:
        raise ValueError("Require Stage-2 training preprocessing (generation_mode=false)")
    # Configuration loading does not instantiate a model or open weight shards.
    model_config = AlpamayoR1Config.from_pretrained(str(checkpoint), local_files_only=True)
    source = instantiate(cfg.data.train_dataset, _convert_="partial", model_config=model_config)
    optimized_source = instantiate(
        cfg.data.train_dataset, _convert_="partial", model_config=model_config,
    )
    if type(source) is not PAIDatasetWithNav or type(optimized_source) is not PAIDatasetWithNav:
        raise TypeError("Data equivalence evidence requires the real PAIDatasetWithNav")
    if len(source) != args.expected_unique_samples or len(optimized_source) != len(source):
        raise ValueError(
            f"Expected {args.expected_unique_samples} unique annotated samples; "
            f"reference={len(source)}, optimized={len(optimized_source)}"
        )
    configured_collate = instantiate(
        cfg.data.collate_fn, _convert_="partial", model_config=model_config,
    )
    cached_collate = partial(collate_fn_from_model_config_cached, model_config=model_config)
    started = time.perf_counter()
    cached, preparation = apply_stage2_training_data(
        optimized_source, preload_max_bytes=args.preload_max_bytes,
    )
    preparation_wall = time.perf_counter() - started
    if (
        not preparation["preload"] or not preparation["drop_raw"]
        or preparation["repeat_factor"] != 1
    ):
        raise RuntimeError("CPU data checker requires preload=1, drop_raw=1, repeat=1")
    forward_signature = inspect.signature(TrainableAlpamayoR1.forward)
    if "image_frames" in forward_signature.parameters:
        raise RuntimeError("image_frames is now an explicit model input; review drop-raw")
    report.update({
        "weights_loaded": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_threads": torch.get_num_threads(), "unique_samples": len(source),
        "checked_unique_samples": 0, "preparation": preparation,
        "preparation_wall_seconds": preparation_wall,
        "resolved_train_dataset_config": OmegaConf.to_container(
            cfg.data.train_dataset, resolve=True,
        ),
        "resolved_collate_config": OmegaConf.to_container(cfg.data.collate_fn, resolve=True),
        "model_config": json.loads(model_config.to_json_string()),
        "checkpoint_config_sha256": hashlib.sha256(
            (checkpoint / "config.json").read_bytes(),
        ).hexdigest(),
        "annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "runtime_source_origins": origins,
        "forward_signature": str(forward_signature),
        "reference_zip_cache_enabled": False,
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "numpy", "physical-ai-av",
        )},
    })
    write_json(args.output, report)
    passed = True
    for index in range(len(source)):
        sample_started = time.perf_counter()
        reference = source[index]
        repeated = source[index]  # Deliberately no reseed between reads.
        if reference is None or repeated is None:
            raise RuntimeError(f"Real source sample {index} could not be decoded")
        deterministic = compare_trees(reference, repeated, torch, np)
        del repeated
        candidate = cached[index]
        if "image_frames" not in reference or "image_frames" in candidate:
            raise RuntimeError(f"Unexpected raw-image availability at sample {index}")
        expected_sample = {key: value for key, value in reference.items() if key != "image_frames"}
        prepared_equal = compare_trees(expected_sample, candidate, torch, np)
        reference_text = reference["tokenized_data"]["text"]
        candidate_text = candidate["tokenized_data"]["text"]
        if not isinstance(reference_text, str) or not isinstance(candidate_text, str):
            raise TypeError("Expected real Qwen preprocessing to produce prompt text strings")
        prompt_equal = reference_text == candidate_text
        baseline_batch = configured_collate([reference])
        candidate_batch = cached_collate([candidate])
        removed = set(baseline_batch) - set(candidate_batch)
        added = set(candidate_batch) - set(baseline_batch)
        raw_only = removed == {"image_frames"} and not added
        batch_equal = compare_trees(
            {key: value for key, value in baseline_batch.items() if key != "image_frames"},
            candidate_batch, torch, np,
        )
        inputs_equal = compare_trees(
            model_inputs(baseline_batch), model_inputs(candidate_batch), torch, np, manifest=True,
        )
        # Exercise both nested tensor/array storage and all mutable containers.
        mutation_counts = mutate_returned_tree(cached[index], torch, np)
        after_mutation = cached[index]
        isolation = compare_trees(expected_sample, after_mutation, torch, np)
        entry = source._samples[index]
        sample_passed = bool(
            deterministic["equal"] and prepared_equal["equal"] and batch_equal["equal"]
            and inputs_equal["equal"] and isolation["equal"] and raw_only and prompt_equal
        )
        passed = passed and sample_passed
        report["samples"].append({
            "index": index, "clip_id": entry.get("clip_id"),
            "t0_relative": entry.get("t0_relative"), "passed": sample_passed,
            "source_read_determinism": deterministic,
            "prepared_sample_and_prompt_equality": prepared_equal,
            "prompt_text": {
                "equal": prompt_equal, "characters": len(reference_text),
                "reference_sha256": hashlib.sha256(reference_text.encode()).hexdigest(),
                "candidate_sha256": hashlib.sha256(candidate_text.encode()).hexdigest(),
            },
            "ordinary_vs_cached_collation": batch_equal,
            "actual_model_input_equality": inputs_equal,
            "raw_frames_only_removed_kwargs": raw_only,
            "removed_batch_keys": sorted(removed), "added_batch_keys": sorted(added),
            "cache_return_mutations": mutation_counts, "cache_return_isolation": isolation,
            "wall_seconds": time.perf_counter() - sample_started,
        })
        report["checked_unique_samples"] = index + 1
        write_json(args.output, report)
        print(
            f"Real PAI sample {index + 1}/{len(source)}: {'PASS' if sample_passed else 'FAIL'}",
            flush=True,
        )
        del reference, candidate, expected_sample, baseline_batch, candidate_batch, after_mutation
        gc.collect()
    return passed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output = args.output.resolve()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # This happens before any heavy imports. The data checker never loads model
    # weights, moves data to CUDA, or invokes any CUDA RNG/compute operation.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.update({
        "APPLY_REPEAT_DATA": "1", "APPLY_PRELOAD": "1", "APPLY_DROP_RAW": "1",
    })
    report: dict[str, Any] = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "all real annotated PAI samples; CPU preprocessing/collation; no model weights",
        "seed": args.seed, "expected_unique_samples": args.expected_unique_samples,
        "comparison": "exact tensor shape/dtype/value and complete scalar/text equality",
        "samples": [],
    }
    write_json(args.output, report)
    try:
        passed = run(args, report)
        report.update(status="passed" if passed else "failed", passed=passed)
        return 0 if passed else 1
    except Exception as exc:
        report.update(
            status="error", passed=False,
            error={
                "type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output, report)
        print(f"CPU data report: {args.output} ({report['status']})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
