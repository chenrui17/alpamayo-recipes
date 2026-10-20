# SPDX-License-Identifier: Apache-2.0
"""Coarse model/trajectory NVTX ranges, enabled only for diagnostic runs."""
from functools import wraps


def instrument_stage2(model):
    import torch

    handles = []
    modules = {"vlm": model.vlm, "expert": model.expert,
               "action_in_proj": model.action_in_proj,
               "action_out_proj": model.action_out_proj,
               "vlm_lm_head": model.vlm.lm_head}
    inner = getattr(model.vlm, "model", None)
    if inner is not None:
        for name in ("visual", "language_model"):
            module = getattr(inner, name, None)
            if module is not None:
                modules["vlm_" + name] = module
    for name, module in modules.items():
        def before(module, inputs, label=name):
            torch.cuda.nvtx.range_push(label)

        def after(module, inputs, output):
            torch.cuda.nvtx.range_pop()

        handles.extend([module.register_forward_pre_hook(before),
                        module.register_forward_hook(after, always_call=True)])

    for name in ("fuse_traj_tokens", "_process_traj_future_training",
                 "_process_position_ids_qwen2_5_vl"):
        original = getattr(model, name)

        @wraps(original)
        def wrapped(*args, _fn=original, _name=name, **kwargs):
            with torch.cuda.nvtx.range(_name):
                return _fn(*args, **kwargs)

        setattr(model, name, wrapped)
    model._stage2_nvtx_handles = handles
