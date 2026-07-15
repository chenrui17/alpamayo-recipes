# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
NVTX markers for Qwen3-VL model using PyTorch forward hooks.

Hook-based approach vs forward-method monkey-patching:
- Hooks fire at module.__call__ level, ABOVE any forward method changes
- Liger's _bind_method_to_module only modifies module.__dict__['forward'],
  which is called FROM __call__ — hooks still fire around the whole __call__
- No risk of double-wrapping or missed wrapping
- Produces proper nested NVTX hierarchy in Nsight Systems
"""
import torch.cuda.nvtx as nvtx

_APPLIED = False
_hook_handles = []   # keep references so handles aren't GC'd


def _hook(module, name):
    """Add NVTX begin/end hooks to module. Idempotent."""
    if getattr(module, '_nvtx_hooked', False):
        return
    object.__setattr__(module, '_nvtx_hooked', True)

    # Use default-arg capture to bind name at definition time (not loop-closure risk)
    def _pre(m, inp, _n=name):
        nvtx.range_push(_n)

    def _post(m, inp, out, _n=name):
        nvtx.range_pop()

    _hook_handles.append(module.register_forward_pre_hook(_pre))
    _hook_handles.append(module.register_forward_hook(_post))


def _wrap_fn(getter, setter, marker):
    """Wrap a module-level function with NVTX markers. Idempotent."""
    fn = getter()
    if getattr(fn, '_nvtx_marker', None) == marker:
        return
    orig = fn
    def _wrapped(*a, _orig=orig, _m=marker, **kw):
        nvtx.range_push(_m)
        out = _orig(*a, **kw)
        nvtx.range_pop()
        return out
    _wrapped._nvtx_marker = marker
    setter(_wrapped)


def apply_nvtx_patches(vlm_model=None):
    """
    vlm_model: Qwen3VLForConditionalGeneration instance.
    Hooks the entire model hierarchy with NVTX markers.
    """
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    from transformers.models.qwen3_vl import modeling_qwen3_vl as m_qvl

    # ── Module-level functions (hooks don't apply; use wrappers) ─────────────
    _wrap_fn(
        lambda: m_qvl.apply_rotary_pos_emb,
        lambda f: setattr(m_qvl, 'apply_rotary_pos_emb', f),
        'rope',
    )
    _wrap_fn(
        lambda: m_qvl.apply_rotary_pos_emb_vision,
        lambda f: setattr(m_qvl, 'apply_rotary_pos_emb_vision', f),
        'rope_vision',
    )

    if vlm_model is None:
        print('[NVTX] Warning: vlm_model=None; only rope wrappers applied', flush=True)
        return

    # ── Qwen3VLForConditionalGeneration ───────────────────────────────────────
    _hook(vlm_model, 'qwen3vl')

    inner = getattr(vlm_model, 'model', None)  # Qwen3VLModel

    # ── Visual encoder ────────────────────────────────────────────────────────
    visual = (getattr(inner, 'visual', None) if inner is not None
              else getattr(vlm_model, 'visual', None))
    if visual is not None:
        _hook(visual, 'visual_encoder')
        for i, blk in enumerate(getattr(visual, 'blocks', [])):
            _hook(blk, f'vis_block_{i:02d}')
            if hasattr(blk, 'norm1'):
                _hook(blk.norm1, 'vis_norm1')
            if hasattr(blk, 'norm2'):
                _hook(blk.norm2, 'vis_norm2')
            if hasattr(blk, 'attn'):
                _hook(blk.attn, 'vis_attn')
            if hasattr(blk, 'mlp'):
                _hook(blk.mlp, 'vis_mlp')

    # ── Language model backbone ───────────────────────────────────────────────
    lm = (getattr(inner, 'language_model', None) if inner is not None
          else getattr(vlm_model, 'language_model', None))
    if lm is None and inner is not None:
        lm = getattr(inner, 'text_model', None)

    if lm is not None:
        _hook(lm, 'language_model')

        # Embed tokens
        if hasattr(lm, 'embed_tokens'):
            _hook(lm.embed_tokens, 'embed_tokens')

        # Decoder layers
        for i, layer in enumerate(getattr(lm, 'layers', [])):
            tag = f'layer_{i:02d}'
            _hook(layer, tag)

            # Input layernorm (pre-attention norm)
            if hasattr(layer, 'input_layernorm'):
                _hook(layer.input_layernorm, 'input_norm')

            # Self-attention
            attn = getattr(layer, 'self_attn', None)
            if attn is not None:
                _hook(attn, 'self_attn')
                # QK norms (Qwen3 has per-head QK-norm)
                if getattr(attn, 'q_norm', None) is not None:
                    _hook(attn.q_norm, 'q_norm')
                if getattr(attn, 'k_norm', None) is not None:
                    _hook(attn.k_norm, 'k_norm')

            # Post-attention layernorm
            if hasattr(layer, 'post_attention_layernorm'):
                _hook(layer.post_attention_layernorm, 'post_attn_norm')

            # MLP (SwiGLU gate_proj + up_proj + activation + down_proj)
            if hasattr(layer, 'mlp'):
                _hook(layer.mlp, 'mlp')

        # Final norm
        if hasattr(lm, 'norm'):
            _hook(lm.norm, 'final_norm')

    # ── LM head ───────────────────────────────────────────────────────────────
    if hasattr(vlm_model, 'lm_head'):
        _hook(vlm_model.lm_head, 'lm_head')

    n = len(_hook_handles) // 2
    print(f'[NVTX] Applied hooks to {n} model components (rope wrappers + hook pairs)', flush=True)
