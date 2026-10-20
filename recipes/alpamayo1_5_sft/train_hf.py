# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from pathlib import Path
import hydra
import hydra.utils as hyu
import torch
from functools import partial
from omegaconf import DictConfig, OmegaConf

from alpamayo_r1.common import logging
from alpamayo.common import misc

from alpamayo1_5_sft.trainer import ReasoningVLA_Trainer
from alpamayo1_5_sft.trainer import TrainingArguments
from alpamayo1_5_sft.performance.apply_optimizations import (
    apply_model_optimizations,
    apply_runtime_optimizations,
    enable_zip_cache,
)
from alpamayo1_5_sft.performance.collate_cache import collate_fn_from_model_config_cached
from alpamayo1_5_sft.performance.perf_utils import perf_plain
from alpamayo1_5_sft.performance.stage2_optimizations import apply_stage2_optimizations
from alpamayo1_5_sft.performance.stage2_data import apply_stage2_training_data
from alpamayo1_5_sft.performance.stage2_trajectory import (
    apply_trajectory_factor_cache,
    trajectory_factor_cache_stats,
)

from alpamayo.common import config_utils
from alpamayo.common import wandb_utils
from alpamayo_r1.common.logging import setup_logging

setup_logging()

logger = logging.RankedLogger("train", rank_zero_only=True)
logger.setLevel("INFO")


@hydra.main(version_base=None, config_path=None, config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training entry point."""
    perf = perf_plain(cfg)
    apply_runtime_optimizations({"optimizations": perf})
    if perf.get("zip_cache", False):
        enable_zip_cache()

    misc.seed_everything(42)

    training_args = TrainingArguments(**OmegaConf.to_container(cfg.trainer, resolve=True))
    logger.info("Configs:\n" + misc.pformat(OmegaConf.to_container(cfg, resolve=True)))

    model = hyu.instantiate(cfg.model, _convert_="partial")
    apply_model_optimizations(model, {"optimizations": perf})
    stage2_applied = apply_stage2_optimizations(model)
    stage2_applied["trajectory"] = apply_trajectory_factor_cache(model)
    if os.environ.get("APPLY_NVTX", "0") == "1":
        from alpamayo1_5_sft.benchmark_h20.nvtx import instrument_stage2

        instrument_stage2(model)
    if int(os.environ.get("RANK", "0")) == 0:
        runtime_path = Path(cfg.paths.output_dir) / "stage2_runtime.json"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(json.dumps({
            "optimizations": stage2_applied,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_vlm": not any(p.requires_grad for p in model.vlm.parameters()),
        }, indent=2) + "\n")

    train_dataset = hyu.instantiate(
        cfg.data.train_dataset, _convert_="partial", model_config=model.config
    )
    train_dataset, stage2_data_report = apply_stage2_training_data(train_dataset)
    if int(os.environ.get("RANK", "0")) == 0:
        data_report_path = Path(cfg.paths.output_dir) / "stage2_data.json"
        data_report_path.parent.mkdir(parents=True, exist_ok=True)
        data_report_path.write_text(json.dumps(stage2_data_report, indent=2) + "\n")
    eval_dataset = hyu.instantiate(
        cfg.data.val_dataset, _convert_="partial", model_config=model.config
    )

    if perf.get("collate_cache", False):
        collate_fn = partial(collate_fn_from_model_config_cached, model_config=model.config)
    else:
        collate_fn = hyu.instantiate(
            cfg.data.collate_fn, _convert_="partial", model_config=model.config
        )

    callbacks = []
    for cb_name, cb_cfg in cfg.callbacks.items():
        logger.info(f"Initializing callback {cb_name}")
        callbacks.append(hyu.instantiate(cb_cfg, _convert_="partial"))

    trainer = ReasoningVLA_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    if "deepspeed" in cfg.trainer and cfg.trainer.deepspeed is not None:
        # We should not cast the forward inputs to bfloat16 because our model is mixed
        # precision and trajectory encoder might require float32 input.
        ds_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config
        ds_config._dtype = torch.float32

    if cfg.get("wandb", None) is not None:
        wandb_utils.init_wandb(**cfg.wandb)

    if trainer.is_world_process_zero():
        config_utils.save_config(
            cfg,
            os.path.join(cfg.paths.output_dir, "config.yaml"),
            resolve_paths=True,
            include_hydra_config=True,
        )

    trainer.train()
    if trainer.is_world_process_zero():
        final_runtime = {
            "trajectory_cache": trajectory_factor_cache_stats(model),
            "vlm_prefix": getattr(model, "_stage2_prefix_report", None),
            "vlm_prefix_totals": getattr(model, "_stage2_prefix_totals", None),
        }
        (Path(cfg.paths.output_dir) / "stage2_final.json").write_text(
            json.dumps(final_runtime, indent=2) + "\n"
        )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
