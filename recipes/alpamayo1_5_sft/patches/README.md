# Liger / diffusion library patches (Stage2 reproduction)

These optimizations patch **installed pip packages** (not this repo tree), so
they are shipped here as idempotent scripts. Run the relevant one against your
own venv **before** launching training with the corresponding env flags.

## patch_swiglu.py  (REQUIRED when APPLY_LIGER_SWIGLU=1, which is the default)

`apply_liger_kernel_to_qwen3_vl(swiglu=True, model=...)` in liger-kernel 0.7.0
is a **no-op** for the SwiGLU path when a concrete `model=` is passed (it only
does the class-level swap, never patches the already-instantiated per-layer
`Qwen3VLTextMLP`). This script fixes `liger_kernel/transformers/monkey_patch.py`
to also do the class-level `Qwen3VLTextMLP -> LigerSwiGLUMLP` swap **and** the
per-decoder-layer `_patch_swiglu_module` on the loaded instance. Idempotent
(re-running is a no-op once patched).

    python patches/patch_swiglu.py

NOTE: the target path is hard-coded to this project venv
(`a1_5_sft/lib/.../liger_kernel/transformers/monkey_patch.py`). Edit the `PATH`
constant at the top to point at your own liger-kernel install before running.
Without this patch, training still runs correctly but SwiGLU stays unfused
(loses the ~1% SwiGLU speedup; rope + rms_norm liger ops are unaffected).

## patch_flce.py  (OPTIONAL / documented negative result)

Enables a custom fused-linear-cross-entropy path. Benchmarked as a **net loss**
on this workload (+1.3 GB memory, ~7.5% slower than plain liger), so training
ships with `fused_linear_cross_entropy=False`. Kept for reference only; not
needed for the reported speedups.
