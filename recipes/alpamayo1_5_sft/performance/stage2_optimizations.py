# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Stage-2 optimizations; installed dependency files are never edited.

Call ``apply_stage2_optimizations(model)`` after loading the model and before
constructing Trainer/DDP. All optimizations are disabled by default. The returned
JSON-compatible report describes the actual patched instances/wrappers.
"""

from __future__ import annotations

import inspect
import os
from types import MethodType
from typing import Any


def stage2_flag(name: str, default: bool = False) -> bool:
    """Read an explicit boolean environment switch without silently accepting typos."""
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean (0/1 or false/true), got {value!r}")


def _text_stack(module: Any) -> Any:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLModel,
        Qwen3VLTextModel,
    )

    if isinstance(module, Qwen3VLForConditionalGeneration):
        return module.model.language_model
    if isinstance(module, Qwen3VLModel):
        return module.language_model
    if isinstance(module, Qwen3VLTextModel):
        return module
    raise TypeError(f"Stage-2 Liger requires dense Qwen3-VL text stacks, got {type(module)}")


def _norms_and_mlps(text_stack: Any) -> tuple[list[Any], list[Any]]:
    norms = [text_stack.norm]
    mlps = []
    for layer in text_stack.layers:
        norms.extend([layer.input_layernorm, layer.post_attention_layernorm])
        for name in ("q_norm", "k_norm"):
            norm = getattr(layer.self_attn, name, None)
            if norm is not None:
                norms.append(norm)
        mlps.append(layer.mlp)
    if not mlps:
        raise ValueError("Stage-2 Liger found no text decoder layers to patch")
    return norms, mlps


def _apply_liger(model: Any, swiglu: bool) -> dict[str, Any]:
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
        from transformers.models.qwen3_vl import modeling_qwen3_vl
    except ImportError as exc:
        raise RuntimeError(
            "APPLY_LIGER_KERNEL=1 requires a compatible liger-kernel installation "
            "with Qwen3-VL support (tested API: liger-kernel 0.7.0, transformers 4.57.1)"
        ) from exc

    # Validate both loaded stacks before changing either one. In Liger 0.7.0 the
    # public Qwen3-VL patch accepts swiglu but does not patch instantiated MLPs.
    # Bind the public LigerSwiGLUMLP.forward to those existing modules ourselves;
    # this preserves their parameters, device placement and state_dict names.
    stacks = {}
    for name in ("vlm", "expert"):
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"Stage-2 Liger requires model.{name}")
        text_stack = _text_stack(module)
        norms, mlps = _norms_and_mlps(text_stack)
        if swiglu:
            activation = getattr(text_stack.config, "hidden_act", None)
            if activation not in {"silu", "swish"}:
                raise ValueError(f"Unsupported SwiGLU activation on {name}: {activation!r}")
            for mlp in mlps:
                if not all(hasattr(mlp, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
                    raise TypeError(f"Unsupported Qwen3-VL MLP layout on {name}: {type(mlp)}")
        stacks[name] = (module, norms, mlps)

    report: dict[str, Any] = {}
    for name, (module, norms, mlps) in stacks.items():
        apply_liger_kernel_to_qwen3_vl(
            rope=True,
            rms_norm=True,
            swiglu=False,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
            model=module,
        )
        if swiglu:
            for mlp in mlps:
                mlp.forward = MethodType(LigerSwiGLUMLP.forward, mlp)

        norm_count = sum(
            getattr(norm.forward, "__func__", None) is LigerRMSNorm.forward for norm in norms
        )
        mlp_count = sum(
            getattr(mlp.forward, "__func__", None) is LigerSwiGLUMLP.forward for mlp in mlps
        )
        if norm_count != len(norms) or (swiglu and mlp_count != len(mlps)):
            raise RuntimeError(
                f"Liger did not patch every requested {name} instance: "
                f"RMSNorm {norm_count}/{len(norms)}, SwiGLU {mlp_count}/{len(mlps)}"
            )
        report[name] = {"rms_norm_instances": norm_count, "swiglu_instances": mlp_count}

    for name in ("apply_rotary_pos_emb", "apply_rotary_pos_emb_vision"):
        function = getattr(modeling_qwen3_vl, name)
        if not getattr(function, "__module__", "").startswith("liger_kernel"):
            raise RuntimeError(f"Liger RoPE patch did not take effect: {name}")
    report["rope"] = True
    report["vision_rope"] = True
    return report


def apply_stage2_optimizations(model: Any) -> dict[str, Any]:
    """Apply opt-in optimizations to a frozen-VLM Stage-2 model.

    ``APPLY_VLM_SKIP_HEAD=1`` skips discarded VLM CE/full-sequence logits in
    ``TrainableAlpamayoR1.forward``. ``APPLY_LIGER_KERNEL=1`` patches loaded VLM
    and expert instances, with SwiGLU enabled unless ``APPLY_LIGER_SWIGLU=0``.
    ``APPLY_COMPILE_EXPERT=1`` compiles the bound ``forward`` methods of the
    expert and action projections after Liger. Original modules and checkpoint
    keys are preserved. Compilation occurs lazily on their first forward, and
    its cost must be reported separately.

    Joint VLM training always keeps the original behavior, even if the process
    inherits Stage-2 switches. An unavailable requested dependency is an error.
    """
    requested = {
        "vlm_skip_head": stage2_flag("APPLY_VLM_SKIP_HEAD"),
        "liger_kernel": stage2_flag("APPLY_LIGER_KERNEL"),
        "compile_expert": stage2_flag("APPLY_COMPILE_EXPERT"),
    }
    report: dict[str, Any] = {name: False for name in requested}
    report["requested"] = requested
    if getattr(model, "cotrain_vlm", True):
        report["skipped_reason"] = "cotrain_vlm"
        return report
    if not any(requested.values()):
        return report

    if any(parameter.requires_grad for parameter in model.vlm.parameters()):
        raise ValueError("Stage-2 optimizations require all VLM parameters to be frozen")

    if requested["vlm_skip_head"]:
        if "logits_to_keep" not in inspect.signature(model.vlm.forward).parameters:
            raise TypeError("Frozen-VLM skip-head requires VLM.forward(logits_to_keep=...)")
        report["vlm_skip_head"] = True

    if requested["liger_kernel"]:
        swiglu = stage2_flag("APPLY_LIGER_SWIGLU", default=True)
        report["liger"] = _apply_liger(model, swiglu)
        report["liger_kernel"] = True
        report["liger_swiglu"] = swiglu

    if requested["compile_expert"]:
        import torch

        if not hasattr(torch, "compile"):
            raise RuntimeError("APPLY_COMPILE_EXPERT=1 requires torch.compile")
        names = ("expert", "action_in_proj", "action_out_proj")
        for name in names:
            if not callable(getattr(getattr(model, name, None), "forward", None)):
                raise ValueError(f"Stage-2 compilation requires model.{name}")
        if not getattr(model, "_stage2_compile_applied", False):
            compiled_forwards = {}
            for name in names:
                module = getattr(model, name)
                original = module.forward
                # torch.compile(module) returns OptimizedModule and registers
                # the original under _orig_mod, changing checkpoint key paths.
                # Compile a bound callable instead and retain the module tree.
                compiled = torch.compile(original, dynamic=True)
                if compiled is original:
                    raise RuntimeError(f"torch.compile was disabled for model.{name}")
                compiled_forwards[name] = compiled
            for name, compiled in compiled_forwards.items():
                getattr(model, name).forward = compiled
            model._stage2_compile_applied = True
        report["compile_expert"] = True
        report["compile"] = {
            "modules": list(names),
            "target": "bound_forward",
            "dynamic": True,
            "execution": "lazy_first_forward",
        }

    return report
