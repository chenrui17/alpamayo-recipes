# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Liger-Kernel vs baseline benchmark for Qwen3-VL-8B-Instruct Stage 1 SFT (nav).

Usage:
    # Baseline (no Liger):
    CUDA_VISIBLE_DEVICES=0 python bench_liger.py --steps 50 --warmup_steps 10 --gradient_checkpointing

    # With Liger (rope + rms_norm + swiglu; fused_linear_cross_entropy disabled because
    #             _compute_next_token_loss requires logits, which the fused kernel skips):
    CUDA_VISIBLE_DEVICES=0 python bench_liger.py --steps 50 --warmup_steps 10 --gradient_checkpointing --liger
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


# ---------------------------------------------------------------------------
# Inline StepTimerCallback (mirrors alpamayo1_sft.benchmark.step_timer)
# ---------------------------------------------------------------------------

class StepTimerCallback(TrainerCallback):
    """Record per-optimizer-step wall time and peak GPU memory."""

    def __init__(
        self,
        warmup_steps: int = 10,
        result_path: str = "benchmark/results/result.json",
        run_name: str = "run",
        stable_tail: int = 40,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.result_path = Path(result_path)
        self.run_name = run_name
        self.stable_tail = stable_tail
        self._step_start: float | None = None
        self._step_times: list[float] = []
        self._current_step = 0

    def on_train_begin(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        torch.cuda.reset_peak_memory_stats()
        self._step_times = []
        self._current_step = 0
        self._step_start = time.perf_counter()

    def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        now = time.perf_counter()
        elapsed = now - self._step_start
        self._current_step += 1
        if self._current_step > self.warmup_steps:
            self._step_times.append(elapsed)
        self._step_start = now

    def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        tail = self._step_times[-self.stable_tail:] if self._step_times else []
        avg_tail = sum(tail) / len(tail) if tail else float("nan")
        avg_all = sum(self._step_times) / len(self._step_times) if self._step_times else float("nan")

        result = {
            "run_name": self.run_name,
            "step_times_sec": self._step_times,
            "avg_step_sec_all": avg_all,
            "avg_step_sec_stable_tail": avg_tail,
            "stable_tail_n": len(tail),
            "peak_gpu_mem_gb": peak_mem_gb,
            "num_measured_steps": len(self._step_times),
            "warmup_steps": self.warmup_steps,
        }
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        with self.result_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        throughput = 1.0 / avg_tail if avg_tail > 0 else 0.0
        print(
            f"\n[BENCHMARK] run={self.run_name}\n"
            f"  avg_step (stable tail {len(tail)} steps): {avg_tail:.3f}s\n"
            f"  avg_step (all {len(self._step_times)} steps):           {avg_all:.3f}s\n"
            f"  throughput:                                {throughput:.3f} steps/s\n"
            f"  peak GPU memory:                           {peak_mem_gb:.2f} GB\n"
            f"  -> {self.result_path}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Liger vs baseline benchmark for A1.5 SFT stage1")
    parser.add_argument("--steps", type=int, default=50, help="Total optimizer steps to run")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Steps to skip when averaging")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable grad checkpoint")
    parser.add_argument("--liger", action="store_true", help="Apply Liger kernel patches")
    parser.add_argument(
        "--liger_fused_ce", action="store_true",
        help="Also apply fused_linear_cross_entropy (WARNING: incompatible with custom loss wrapper "
             "because _compute_next_token_loss requires logits; use only if model is modified)"
    )
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    args = parser.parse_args()

    run_name = "liger" if args.liger else "baseline"
    result_path = args.result_path or f"benchmark/results/{run_name}.json"

    print(f"\n{'='*70}")
    print(f"  Benchmark: {run_name}")
    print(f"  steps={args.steps}  warmup={args.warmup_steps}  gc={args.gradient_checkpointing}")
    print(f"  liger={args.liger}  liger_fused_ce={args.liger_fused_ce}")
    print(f"  batch_size={args.batch_size}  grad_accum={args.grad_accum}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

    print("[1/4] Loading model from /raid/charlie/Alpamayo-1.5-10B-A1-format ...", flush=True)
    model = TrainableReasoningVLA.from_alpamayo_checkpoint(
        checkpoint_path="/raid/charlie/Alpamayo-1.5-10B-A1-format",
        vlm_name_or_path="Qwen/Qwen3-VL-8B-Instruct",
    )

    # ------------------------------------------------------------------
    # 2. Apply Liger kernel patches (AFTER model load)
    # ------------------------------------------------------------------
    liger_fused_ce_applied = False
    if args.liger:
        print("[2/4] Applying Liger kernel patches ...", flush=True)
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl
        # fused_linear_cross_entropy=True is incompatible with _compute_next_token_loss
        # because that function needs outputs.logits (set to None by fused CE kernel).
        # We apply fused_linear_cross_entropy only if explicitly requested.
        fused_ce = args.liger_fused_ce
        apply_liger_kernel_to_qwen3_vl(
            rope=True,
            rms_norm=True,
            fused_linear_cross_entropy=fused_ce,
            swiglu=True,
            model=model.vlm,
        )
        liger_fused_ce_applied = fused_ce
        applied = ["rope", "rms_norm", "swiglu"] + (["fused_linear_ce"] if fused_ce else [])
        print(f"  Applied: {', '.join(applied)}", flush=True)
        if not fused_ce:
            print(
                "  NOTE: fused_linear_cross_entropy SKIPPED - incompatible with custom loss\n"
                "        wrapper (_compute_next_token_loss needs outputs.logits, which the\n"
                "        fused CE kernel does not materialise when skip_logits=True).\n"
                "        Use --liger_fused_ce to override (requires model modifications).",
                flush=True,
            )
    else:
        print("[2/4] No Liger patches (baseline run)", flush=True)

    if args.gradient_checkpointing:
        print("  Enabling gradient checkpointing ...", flush=True)
        model.gradient_checkpointing_enable({"use_reentrant": False})

    # ------------------------------------------------------------------
    # 3. Build dataset and collate_fn (using the actual Stage-1 nav pipeline)
    # ------------------------------------------------------------------
    print("[3/4] Building dataset (Stage-1 nav) ...", flush=True)

    from alpamayo.data.pai_nav import PAIDatasetWithNav
    from alpamayo.processor.qwen_processor import (
        get_preprocess_data_fn_from_model_config,
        collate_fn_from_model_config,
    )

    # Build the VLA pre-processing function (same config as sft_stage1_nav.yaml)
    vla_preprocess_args = {
        "_target_": "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
        "chat_template_version": "r1_5",
        "components_order": ["image", "traj_history", "route", "prompt", "traj_future"],
        "components_prompt": ["traj_future"],
        "label_components": ["traj_future"],
        "include_camera_ids": True,
        "include_frame_nums": True,
        "generation_mode": False,
    }

    train_dataset = PAIDatasetWithNav(
        local_dir="/raid/charlie/pai_dataset/",
        annotations_path=(
            "/raid/charlie/alpamayo/alpamayo1.5/notebooks/nav_demo_samples.json"
        ),
        chunk_ids=[
            214, 224, 276, 317, 420, 727, 728, 968, 982,
            1519, 1657, 1984, 2277, 2368, 2372, 2447, 2599, 2634, 2868,
        ],
        use_default_keyframe=True,
        model_config=model.config,
        vla_preprocess_args=vla_preprocess_args,
    )
    print(f"  Dataset size: {len(train_dataset)} samples", flush=True)

    collate_fn = partial(
        collate_fn_from_model_config,
        model_config=model.config,
        chat_template_version="r1_5",
    )

    # ------------------------------------------------------------------
    # 4. Run training with HF Trainer
    # ------------------------------------------------------------------
    print("[4/4] Running training loop ...", flush=True)

    gc_kwargs: dict[str, Any] = {}
    if args.gradient_checkpointing:
        gc_kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}

    training_args = TrainingArguments(
        output_dir=f"benchmark/runs/{run_name}",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.steps + args.warmup_steps,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        save_strategy="no",
        logging_steps=args.steps + args.warmup_steps + 1,  # suppress default logging
        warmup_steps=0,
        lr_scheduler_type="constant",
        learning_rate=1e-5,
        **gc_kwargs,
    )

    timer_cb = StepTimerCallback(
        warmup_steps=args.warmup_steps,
        result_path=result_path,
        run_name=run_name,
        stable_tail=args.steps - args.warmup_steps,
    )

    from alpamayo1_5_sft.trainer import ReasoningVLA_Trainer

    trainer = ReasoningVLA_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[timer_cb],
    )

    # Pre-warm torch.linalg.cholesky to avoid lazy-wrapper init error in PyTorch 2.8
    print('[pre-warm] Initializing torch.linalg.cholesky CUDA backend...', flush=True)
    import torch as _torch
    _d = _torch.eye(4, device='cuda', dtype=_torch.float32)
    _ = _torch.linalg.cholesky(_d)
    _torch.cuda.synchronize()
    del _d, _
    print('[pre-warm] Done.', flush=True)

    trainer.train()

    print(f"\n[DONE] liger_fused_ce_applied={liger_fused_ce_applied}", flush=True)


if __name__ == "__main__":
    main()
