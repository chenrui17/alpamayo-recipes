# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative, opt-in frozen Qwen3-VL prefix preparation for Stage 2.

The causal VLM KV states used by the expert end at ``future_start``. For the
restricted input/configuration checked here, text after that token cannot alter
the prefix KV, and removing a plain-text suffix preserves default MRoPE deltas.
This is an experimental candidate: verify loss and gradients on the real model
and workload before claiming equivalence. This helper never caches KV states.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

from .stage2_optimizations import stage2_flag


def maybe_trim_future_prefix(
    model: Any, input_ids: Any, tokenized_data: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Return VLM-only ids/kwargs and a report, preserving every caller input.

    ``APPLY_VLM_PREFIX_ONLY`` defaults to off. The disabled path reads tensor
    shape metadata only; it imports no torch/transformers and does no GPU work.
    Unsupported enabled cases return the exact original ids and kwargs objects.
    An applied result keeps the inclusive ``future_start`` prefix and slices only
    its attention mask. Visual tensors and grids retain their original identity.
    The caller must retain its original full ids for the downstream crop/targets.
    """
    shape = getattr(input_ids, "shape", ())
    before = int(shape[1]) if len(shape) == 2 else None
    requested = stage2_flag("APPLY_VLM_PREFIX_ONLY")
    report = {
        "requested": requested,
        "applied": False,
        "tokens_before": before,
        "tokens_after": before,
        "trimmed_tokens": 0,
        "fallback_reason": "disabled",
    }

    def fallback(reason: str) -> tuple[Any, Any, dict[str, Any]]:
        report["fallback_reason"] = reason
        return input_ids, tokenized_data, report

    if not requested:
        return input_ids, tokenized_data, report
    if getattr(model, "cotrain_vlm", True):
        return fallback("joint_vlm_training")
    if not stage2_flag("APPLY_VLM_SKIP_HEAD"):
        return fallback("skip_head_required")
    vlm = getattr(model, "vlm", None)
    if vlm is None or not callable(getattr(vlm, "parameters", None)):
        return fallback("missing_vlm")
    if any(parameter.requires_grad for parameter in vlm.parameters()):
        return fallback("unfrozen_vlm")
    if not isinstance(tokenized_data, Mapping):
        return fallback("kwargs_not_mapping")
    for name in ("position_ids", "cache_position", "rope_deltas"):
        if name in tokenized_data:
            return fallback(f"explicit_{name}")
    if tokenized_data.get("past_key_values") is not None:
        return fallback("existing_past_key_values")
    if tokenized_data.get("inputs_embeds") is not None:
        return fallback("explicit_inputs_embeds")
    # Unknown per-token inputs or attention controls could invalidate truncation.
    supported = {
        "attention_mask", "pixel_values", "pixel_values_videos", "image_grid_thw",
        "video_grid_thw", "second_per_grid_ts",
    }
    unknown = set(tokenized_data) - supported
    if unknown:
        return fallback("unsupported_kwargs:" + ",".join(sorted(map(str, unknown))))
    try:
        import torch
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        return fallback("qwen3_vl_dependency_unavailable")
    if not isinstance(vlm, Qwen3VLForConditionalGeneration):
        return fallback("unsupported_vlm_class")
    config = vlm.config
    if getattr(config, "model_type", None) != "qwen3_vl":
        return fallback("unsupported_vlm_config")
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return fallback("missing_text_config")
    for name in ("rope_scaling", "rope_parameters"):
        rope = getattr(text_config, name, None)
        if rope is not None:
            if not isinstance(rope, Mapping):
                return fallback("unsupported_rope_config")
            if rope.get("rope_type", rope.get("type", "default")) != "default":
                return fallback("nondefault_rope")
    def zero_dropout(value: Any) -> bool:
        return isinstance(value, Real) and float(value) == 0.0

    if not zero_dropout(getattr(text_config, "attention_dropout", None)):
        return fallback("nonzero_attention_dropout")
    vision_config = getattr(config, "vision_config", None)
    if not zero_dropout(getattr(vision_config, "attention_dropout", 0.0)):
        return fallback("nonzero_attention_dropout")
    for module in vlm.modules():
        if getattr(module, "rope_type", "default") != "default":
            return fallback("nondefault_rope")
        if not zero_dropout(getattr(module, "attention_dropout", 0.0)):
            return fallback("nonzero_attention_dropout")
        if isinstance(module, torch.nn.Dropout) and module.p != 0.0:
            return fallback("nonzero_dropout")
    if not torch.is_tensor(input_ids) or input_ids.dtype not in (torch.int32, torch.int64):
        return fallback("unsupported_input_ids")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        return fallback("requires_nonempty_batch_one")
    if input_ids.device.type == "meta":
        return fallback("meta_input_ids")
    mask = tokenized_data.get("attention_mask")
    if not torch.is_tensor(mask) or tuple(mask.shape) != tuple(input_ids.shape):
        return fallback("requires_full_attention_mask")
    if mask.device != input_ids.device:
        return fallback("attention_mask_device_mismatch")
    if not bool(torch.all(mask == 1).item()):
        return fallback("padding_or_nonfull_attention_mask")
    traj_ids = getattr(getattr(model, "config", None), "traj_token_ids", {})
    future_start = traj_ids.get("future_start") if isinstance(traj_ids, Mapping) else None
    if type(future_start) is not int:
        return fallback("missing_future_start_token_id")
    positions = (input_ids == future_start).nonzero(as_tuple=False)
    if positions.shape[0] != 1:
        return fallback("requires_exactly_one_future_start")
    end = int(positions[0, 1].item()) + 1
    if end == before:
        return fallback("no_suffix_to_trim")
    visual_ids = [getattr(config, name, None) for name in (
        "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
    )]
    if any(type(value) is not int for value in visual_ids):
        return fallback("missing_visual_token_ids")
    forbidden = torch.tensor(visual_ids, device=input_ids.device, dtype=input_ids.dtype)
    if bool(torch.isin(input_ids[:, end:], forbidden).any().item()):
        return fallback("visual_tokens_after_future_start")
    vlm_kwargs = dict(tokenized_data)
    vlm_kwargs["attention_mask"] = mask[:, :end]
    report.update(
        applied=True, tokens_after=end, trimmed_tokens=before - end, fallback_reason=None,
    )
    return input_ids[:, :end], vlm_kwargs, report
