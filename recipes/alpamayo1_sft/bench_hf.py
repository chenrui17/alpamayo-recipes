# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark entry point: fixed-step runs with stable-tail timing."""

from __future__ import annotations

import hydra
import hydra.utils as hyu
import torch
from omegaconf import DictConfig, OmegaConf

from alpamayo_r1.common import logging
from alpamayo.common import misc

from alpamayo1_sft.trainer import ReasoningVLA_Trainer, TrainingArguments
from alpamayo1_sft.benchmark.profile_callback import ProfileRangeCallback
from alpamayo1_sft.benchmark.step_timer import StepTimerCallback
from alpamayo1_sft.benchmark.apply_optimizations import (
    apply_model_optimizations,
    apply_runtime_optimizations,
    enable_zip_cache,
)
from alpamayo1_sft.benchmark.load_cache import enable_load_cache, preload_load_cache
from alpamayo1_sft.benchmark.preprocess_cache import enable_preprocess_cache, preload_preprocess_cache
from alpamayo1_sft.benchmark.dataloader_timing import wrap_dataset_for_timing
from alpamayo1_sft.benchmark.dataloader_timing_callback import DataloaderTimingCallback
from alpamayo1_sft.benchmark.perf_utils import build_collate_fn, perf_plain
from alpamayo1_sft.benchmark.trainer_nvtx import patch_trainer_nvtx
from alpamayo1_sft.benchmark.trainer_timing import patch_trainer_timing
from alpamayo_r1.common.logging import setup_logging

TRAINER_OVERRIDE_KEYS = {
    "dataloader_num_workers",
    "dataloader_pin_memory",
    "dataloader_persistent_workers",
    "dataloader_prefetch_factor",
    "gradient_checkpointing",
    "gradient_accumulation_steps",
}

setup_logging()

logger = logging.RankedLogger("bench", rank_zero_only=True)
logger.setLevel("INFO")


def _to_plain(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)  # type: ignore[return-value]
    return dict(obj)


@hydra.main(version_base=None, config_path=None, config_name="config")
def bench(cfg: DictConfig) -> None:
    bench_cfg = cfg.get("benchmark") or {}
    bench_plain = _to_plain(bench_cfg)
    run_name = bench_plain.get("run_name", "baseline")
    max_steps = int(bench_plain.get("max_steps", 50))
    stable_tail = int(bench_plain.get("stable_tail", 40))
    result_path = bench_plain.get("result_path", f"benchmark/results/{run_name}.json")

    perf = perf_plain(cfg)
    if bench_plain.get("nvtx_markers", False):
        perf["nvtx_markers"] = True
    if bench_plain.get("dataloader_timing", False):
        perf["dataloader_timing"] = True
    runtime_applied = apply_runtime_optimizations({"optimizations": perf})
    if perf.get("zip_cache", False):
        enable_zip_cache()

    misc.seed_everything(42)

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs["max_steps"] = max_steps
    trainer_kwargs["num_train_epochs"] = 100000
    trainer_kwargs["warmup_steps"] = 0
    trainer_kwargs["logging_steps"] = max_steps + 1
    trainer_kwargs["save_steps"] = max_steps + 1
    trainer_kwargs["report_to"] = "none"

    for key in TRAINER_OVERRIDE_KEYS:
        if key in bench_plain:
            trainer_kwargs[key] = bench_plain[key]

    training_args = TrainingArguments(**trainer_kwargs)

    model = hyu.instantiate(cfg.model, _convert_="partial")
    model_applied = apply_model_optimizations(model, {"optimizations": perf})

    train_dataset = hyu.instantiate(
        cfg.data.train_dataset, _convert_="partial", model_config=model.config
    )
    if perf.get("preprocess_cache", False):
        enable_preprocess_cache(train_dataset)
    if perf.get("load_cache", False):
        train_dataset = enable_load_cache(train_dataset)
        if perf.get("preload_load_cache", True):
            preload_load_cache(train_dataset)
    elif perf.get("preprocess_cache", False) and perf.get("preload_preprocess_cache", True):
        preload_preprocess_cache(train_dataset)
    if perf.get("dataloader_timing", False):
        log_every = int(bench_plain.get("dataloader_timing_sample_log_every", 20))
        train_dataset = wrap_dataset_for_timing(train_dataset, log_every=log_every)
    eval_dataset = hyu.instantiate(
        cfg.data.val_dataset, _convert_="partial", model_config=model.config
    )
    collate_fn = build_collate_fn(cfg, model, perf)

    timer_cb = StepTimerCallback(
        output_path=result_path,
        run_name=run_name,
        config={
            "benchmark": bench_plain,
            "runtime": runtime_applied,
            "model": model_applied,
            "optimizations": perf,
        },
        stable_tail=stable_tail,
    )

    callbacks = [timer_cb]
    if perf.get("dataloader_timing", False):
        timing_path = bench_plain.get(
            "dataloader_timing_path",
            f"benchmark/profiles/{run_name}_dataloader_timing.json",
        )
        callbacks.append(
            DataloaderTimingCallback(
                output_path=timing_path,
                log_start_step=int(bench_plain.get("dataloader_timing_start", 5)),
                log_every=int(bench_plain.get("dataloader_timing_log_every", 1)),
                meta={"run_name": run_name, "optimizations": perf},
            )
        )
    profile_start = bench_plain.get("profile_start")
    profile_end = bench_plain.get("profile_end")
    if profile_start is not None and profile_end is not None:
        callbacks.append(
            ProfileRangeCallback(
                profile_start=int(profile_start),
                profile_end=int(profile_end),
            )
        )

    trainer = ReasoningVLA_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    if perf.get("nvtx_markers", False):
        patch_trainer_nvtx(trainer)
    if perf.get("dataloader_timing", False):
        patch_trainer_timing(trainer)

    if "deepspeed" in cfg.trainer and cfg.trainer.deepspeed is not None and training_args.deepspeed:
        ds_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config
        ds_config._dtype = torch.float32

    trainer.train()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    bench()
