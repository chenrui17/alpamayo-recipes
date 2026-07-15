# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collate with per-phase wall-clock timing."""

from __future__ import annotations

import time
from typing import Any

import torch

from alpamayo.processor.qwen_processor import QwenProcessor, basic_collation_fn
from alpamayo.utils.get_label_mask import get_label_mask, get_role_eos_mask
from alpamayo1_sft.benchmark.collate_cache import _COLLATE_PROCESSORS
from alpamayo1_sft.benchmark.dataloader_timing import collect_sample_timings


def collate_fn_with_timing(
    processor: QwenProcessor,
    data: list[dict[str, Any]],
    padding_side: str = "left",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    batch_timing: dict[str, Any] = dict(collect_sample_timings(data))

    t1 = time.perf_counter()
    batched_data: dict[str, Any] = basic_collation_fn(data, unstackable_keys=["image_frames"])
    t2 = time.perf_counter()
    batch_timing["collate.basic"] = t2 - t1

    tokenized_data = {}
    t3 = time.perf_counter()
    for k in batched_data["tokenized_data"][0].keys():
        if k not in ["text"]:
            tokenized_data[k] = torch.cat([row[k] for row in batched_data["tokenized_data"]])
    t4 = time.perf_counter()
    batch_timing["collate.stack"] = t4 - t3

    t5 = time.perf_counter()
    batch_text = [instance["text"] for instance in batched_data["tokenized_data"]]
    processed_inputs = processor.processor.tokenizer(
        batch_text, return_tensors="pt", padding_side=padding_side, padding=True
    )
    tokenized_data.update(processed_inputs)
    batched_data["tokenized_data"] = tokenized_data
    t6 = time.perf_counter()
    batch_timing["collate.tokenizer"] = t6 - t5

    label_components = batched_data["label_components"][0]
    generation_mode = batched_data["generation_mode"][0]

    t7 = time.perf_counter()
    if not generation_mode:
        batched_data["labels_mask"] = get_label_mask(
            input_ids=tokenized_data["input_ids"],
            tokenizer=processor.processor.tokenizer,
            label_components=label_components,
        )
        eos_mask_assistant = get_role_eos_mask(
            input_ids=tokenized_data["input_ids"],
            tokenizer=processor.processor.tokenizer,
            bos_token="<|im_start|>",
            role="assistant",
        )
        batched_data["labels_mask"] |= eos_mask_assistant
    else:
        batched_data["labels_mask"] = torch.zeros_like(
            tokenized_data["input_ids"],
            dtype=torch.bool,
            device=tokenized_data["input_ids"].device,
        )
    t8 = time.perf_counter()
    batch_timing["collate.label_mask"] = t8 - t7
    batch_timing["collate.total"] = time.perf_counter() - t0
    batched_data["_batch_timing"] = batch_timing
    return batched_data


def collate_fn_from_model_config_cached_timed(
    data: list[dict[str, Any]],
    model_config=None,
    padding_side: str = "left",
    include_camera_ids: bool = False,
    include_frame_nums: bool = False,
    chat_template_version: str = "r1",
) -> dict[str, Any]:
    key = (
        model_config.vlm_name_or_path,
        model_config.traj_vocab_size,
        getattr(model_config, "min_pixels", None),
        getattr(model_config, "max_pixels", None),
        include_camera_ids,
        include_frame_nums,
        chat_template_version,
        padding_side,
    )
    if key not in _COLLATE_PROCESSORS:
        _COLLATE_PROCESSORS[key] = QwenProcessor(
            vlm_name_or_path=model_config.vlm_name_or_path,
            traj_vocab_size=model_config.traj_vocab_size,
            min_pixels=getattr(model_config, "min_pixels", None),
            max_pixels=getattr(model_config, "max_pixels", None),
            include_camera_ids=include_camera_ids,
            include_frame_nums=include_frame_nums,
            chat_template_version=chat_template_version,
        )
    return collate_fn_with_timing(_COLLATE_PROCESSORS[key], data, padding_side=padding_side)
