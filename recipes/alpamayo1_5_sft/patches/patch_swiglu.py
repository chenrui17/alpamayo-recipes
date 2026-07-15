#!/usr/bin/env python3
"""Fix apply_liger_kernel_to_qwen3_vl so `swiglu=True` is honored (was a no-op).
Mirrors apply_liger_kernel_to_qwen3: class-level Qwen3VLTextMLP -> LigerSwiGLUMLP,
plus per-decoder-layer _patch_swiglu_module on the loaded instance.
"""
import sys
import py_compile

PATH = ("/raid/charlie/alpamayo/alpamayo-recipes/recipes/alpamayo1_5_sft/"
        "a1_5_sft/lib/python3.12/site-packages/liger_kernel/transformers/monkey_patch.py")

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

if "modeling_qwen3_vl.Qwen3VLTextMLP = LigerSwiGLUMLP" in src:
    print("Already patched; nothing to do.")
    sys.exit(0)

orig = src

# --- 1. class-level swiglu replacement (right after rms_norm class swap) ---
cls_old = (
    "    if rms_norm:\n"
    "        modeling_qwen3_vl.Qwen3VLTextRMSNorm = LigerRMSNorm\n"
    "\n"
    "    if cross_entropy:\n"
)
assert src.count(cls_old) == 1, "qwen3_vl rms_norm class-swap anchor not unique/found"
cls_new = (
    "    if rms_norm:\n"
    "        modeling_qwen3_vl.Qwen3VLTextRMSNorm = LigerRMSNorm\n"
    "\n"
    "    if swiglu:\n"
    "        modeling_qwen3_vl.Qwen3VLTextMLP = LigerSwiGLUMLP\n"
    "\n"
    "    if cross_entropy:\n"
)
src = src.replace(cls_old, cls_new, 1)

# --- 2. instance-level: honor swiglu + gate rms_norm parts ---
inst_old = (
    "    if model is not None and rms_norm:\n"
    "        if isinstance(model, Qwen3VLForConditionalGeneration):\n"
    "            text_model: Qwen3VLTextModel = model.model.language_model\n"
    "        elif isinstance(model, Qwen3VLModel):\n"
    "            text_model: Qwen3VLTextModel = model.language_model\n"
    "        elif isinstance(model, Qwen3VLTextModel):\n"
    "            text_model = model\n"
    "        else:\n"
    "            raise TypeError(\n"
    "                f\"Unsupported Qwen3VL model type. `model` must be `Qwen3VLForConditionalGeneration`, `Qwen3VLModel` or `Qwen3VLTextModel`. Got: {type(model)}\"\n"
    "            )\n"
    "\n"
    "        _patch_qwen3_vl_rms_norm = partial(_patch_rms_norm_module, offset=0.0, casting_mode=\"llama\")\n"
    "\n"
    "        if text_model is not None:\n"
    "            _patch_qwen3_vl_rms_norm(text_model.norm)\n"
    "            for decoder_layer in text_model.layers:\n"
    "                _patch_qwen3_vl_rms_norm(decoder_layer.input_layernorm)\n"
    "                _patch_qwen3_vl_rms_norm(decoder_layer.post_attention_layernorm)\n"
    "                self_attn = getattr(decoder_layer, \"self_attn\", None)\n"
    "                if self_attn is not None:\n"
    "                    if hasattr(self_attn, \"q_norm\") and self_attn.q_norm is not None:\n"
    "                        _patch_qwen3_vl_rms_norm(self_attn.q_norm)\n"
    "                    if hasattr(self_attn, \"k_norm\") and self_attn.k_norm is not None:\n"
    "                        _patch_qwen3_vl_rms_norm(self_attn.k_norm)\n"
)
assert src.count(inst_old) == 1, "qwen3_vl instance rms_norm block anchor not unique/found"
inst_new = (
    "    if model is not None and (rms_norm or swiglu):\n"
    "        if isinstance(model, Qwen3VLForConditionalGeneration):\n"
    "            text_model: Qwen3VLTextModel = model.model.language_model\n"
    "        elif isinstance(model, Qwen3VLModel):\n"
    "            text_model: Qwen3VLTextModel = model.language_model\n"
    "        elif isinstance(model, Qwen3VLTextModel):\n"
    "            text_model = model\n"
    "        else:\n"
    "            raise TypeError(\n"
    "                f\"Unsupported Qwen3VL model type. `model` must be `Qwen3VLForConditionalGeneration`, `Qwen3VLModel` or `Qwen3VLTextModel`. Got: {type(model)}\"\n"
    "            )\n"
    "\n"
    "        _patch_qwen3_vl_rms_norm = partial(_patch_rms_norm_module, offset=0.0, casting_mode=\"llama\")\n"
    "\n"
    "        if text_model is not None:\n"
    "            if rms_norm:\n"
    "                _patch_qwen3_vl_rms_norm(text_model.norm)\n"
    "            for decoder_layer in text_model.layers:\n"
    "                if rms_norm:\n"
    "                    _patch_qwen3_vl_rms_norm(decoder_layer.input_layernorm)\n"
    "                    _patch_qwen3_vl_rms_norm(decoder_layer.post_attention_layernorm)\n"
    "                    self_attn = getattr(decoder_layer, \"self_attn\", None)\n"
    "                    if self_attn is not None:\n"
    "                        if hasattr(self_attn, \"q_norm\") and self_attn.q_norm is not None:\n"
    "                            _patch_qwen3_vl_rms_norm(self_attn.q_norm)\n"
    "                        if hasattr(self_attn, \"k_norm\") and self_attn.k_norm is not None:\n"
    "                            _patch_qwen3_vl_rms_norm(self_attn.k_norm)\n"
    "                if swiglu:\n"
    "                    _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)\n"
)
src = src.replace(inst_old, inst_new, 1)

assert src != orig, "no changes applied"

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

py_compile.compile(PATH, doraise=True)
print("Patched apply_liger_kernel_to_qwen3_vl (swiglu now honored) and byte-compiled OK.")
