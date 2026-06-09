# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark entry point: fixed 15-step runs with stable-tail timing."""

from __future__ import annotations

import os

import hydra
import hydra.utils as hyu
import torch
from omegaconf import DictConfig, OmegaConf

from alpamayo_r1.common import logging
from alpamayo.common import misc

from alpamayo1_5_sft.trainer import ReasoningVLA_Trainer, TrainingArguments
from alpamayo1_5_sft.benchmark.apply_optimizations import (
    apply_model_optimizations,
    apply_runtime_optimizations,
    enable_zip_cache,
)
from alpamayo1_5_sft.benchmark.collate_cache import collate_fn_from_model_config_cached
from alpamayo1_5_sft.benchmark.sample_cache import InMemorySampleCache
from alpamayo1_5_sft.benchmark.step_timer import StepTimerCallback
from alpamayo_r1.common.logging import setup_logging

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

setup_logging()

logger = logging.RankedLogger("bench", rank_zero_only=True)
logger.setLevel("INFO")


def _build_collate(cfg: DictConfig, model):
    bench = cfg.get("benchmark") or {}
    opt = bench.get("optimizations") or {}
    if opt.get("collate_cache", False):
        from functools import partial

        return partial(collate_fn_from_model_config_cached, model_config=model.config)
    return hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)


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
    max_steps = int(bench_plain.get("max_steps", 15))
    stable_tail = int(bench_plain.get("stable_tail", 10))
    result_path = bench_plain.get("result_path", f"benchmark/results/{run_name}.json")

    runtime_applied = apply_runtime_optimizations(bench_plain)
    opt = bench_plain.get("optimizations") or {}
    if opt.get("zip_cache", False):
        enable_zip_cache()

    misc.seed_everything(42)

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs["max_steps"] = max_steps
    trainer_kwargs["num_train_epochs"] = 1
    trainer_kwargs["warmup_steps"] = 0
    trainer_kwargs["logging_steps"] = max_steps + 1
    trainer_kwargs["save_steps"] = max_steps + 1
    trainer_kwargs["report_to"] = "none"

    # Optional benchmark-only trainer overrides (also settable via trainer.* Hydra overrides)
    for key in TRAINER_OVERRIDE_KEYS:
        if key in bench_plain:
            trainer_kwargs[key] = bench_plain[key]

    if opt.get("no_deepspeed", False):
        trainer_kwargs["deepspeed"] = None

    training_args = TrainingArguments(**trainer_kwargs)

    model = hyu.instantiate(cfg.model, _convert_="partial")
    model_applied = apply_model_optimizations(model, bench_plain)

    train_dataset = hyu.instantiate(
        cfg.data.train_dataset, _convert_="partial", model_config=model.config
    )
    if opt.get("preload_dataset_cache", False):
        train_dataset = InMemorySampleCache(train_dataset, preload=True)
    eval_dataset = hyu.instantiate(
        cfg.data.val_dataset, _convert_="partial", model_config=model.config
    )
    collate_fn = _build_collate(cfg, model)

    timer_cb = StepTimerCallback(
        output_path=result_path,
        run_name=run_name,
        config={
            "runtime": runtime_applied,
            "model": model_applied,
            "optimizations": opt,
            "trainer_overrides": {
                k: trainer_kwargs[k]
                for k in trainer_kwargs
                if k.startswith("dataloader_") or k.startswith("torch_compile") or k in (
                    "gradient_checkpointing",
                    "gradient_accumulation_steps",
                )
            },
        },
        stable_tail=stable_tail,
    )

    trainer = ReasoningVLA_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[timer_cb],
    )

    if "deepspeed" in cfg.trainer and cfg.trainer.deepspeed is not None and training_args.deepspeed:
        ds_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config
        ds_config._dtype = torch.float32

    trainer.train()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    bench()
