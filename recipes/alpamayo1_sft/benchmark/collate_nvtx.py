# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collate with NVTX sub-ranges for CPU-side pipeline diagnosis."""

from __future__ import annotations

from typing import Any

import torch

from alpamayo.processor.qwen_processor import QwenProcessor, basic_collation_fn
from alpamayo.utils.get_label_mask import get_label_mask, get_role_eos_mask
from alpamayo1_sft.benchmark.collate_cache import _COLLATE_PROCESSORS
from alpamayo1_sft.benchmark.nvtx_markers import local_rank, nvtx_range


def collate_fn_with_nvtx(
    processor: QwenProcessor,
    data: list[dict[str, Any]],
    padding_side: str = "left",
) -> dict[str, Any]:
    rank = local_rank()
    prefix = f"rank{rank}"

    with nvtx_range(f"{prefix}.collate_fn.inner"):
        with nvtx_range(f"{prefix}.collate.basic_collation"):
            batched_data: dict[str, Any] = basic_collation_fn(
                data, unstackable_keys=["image_frames"]
            )

        tokenized_data = {}
        with nvtx_range(f"{prefix}.collate.stack_tokenized"):
            for k in batched_data["tokenized_data"][0].keys():
                if k not in ["text"]:
                    tokenized_data[k] = torch.cat(
                        [row[k] for row in batched_data["tokenized_data"]]
                    )

        with nvtx_range(f"{prefix}.collate.tokenizer"):
            batch_text = [instance["text"] for instance in batched_data["tokenized_data"]]
            processed_inputs = processor.processor.tokenizer(
                batch_text, return_tensors="pt", padding_side=padding_side, padding=True
            )
            tokenized_data.update(processed_inputs)
            batched_data["tokenized_data"] = tokenized_data

        label_components = batched_data["label_components"][0]
        assert all(l_i == label_components for l_i in batched_data["label_components"]), (
            "label_components is not the same for all instances in the batch."
        )

        generation_mode = batched_data["generation_mode"][0]
        assert all(g_i == generation_mode for g_i in batched_data["generation_mode"]), (
            "generation_mode is not the same for all instances in the batch."
        )

        if not generation_mode:
            with nvtx_range(f"{prefix}.collate.label_mask"):
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

    return batched_data


def collate_fn_from_model_config_cached_nvtx(
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
    return collate_fn_with_nvtx(
        _COLLATE_PROCESSORS[key], data, padding_side=padding_side
    )
