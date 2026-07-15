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

"""Shared FSDP2 sharding helpers for Alpamayo Cosmos-RL model wrappers."""

from __future__ import annotations

from typing import Callable

import torch
from cosmos_rl.utils.logging import logger


def iter_blocks(blocks: object):
    """Iterate over ModuleDict, ModuleList, Sequential, or plain sequences."""
    from torch import nn

    if isinstance(blocks, nn.ModuleDict):
        yield from blocks.items()
        return
    if isinstance(blocks, (nn.ModuleList, nn.Sequential)):
        yield from enumerate(blocks)
        return
    if isinstance(blocks, (list, tuple)):
        yield from enumerate(blocks)
        return
    yield 0, blocks


def find_first_attr_chain(root: object, chains: list[list[str]]):
    """Walk attribute chains on *root*, return the first that resolves."""
    for chain in chains:
        cur = root
        ok = True
        for name in chain:
            cur = getattr(cur, name, None)
            if cur is None:
                ok = False
                break
        if ok:
            return cur
    return None


def check_parallelism_preconditions(parallel_dims, config, model_name: str) -> None:
    assert not parallel_dims.tp_enabled, f"TP not supported for {model_name}"
    assert not parallel_dims.cp_enabled, f"CP not supported for {model_name}"
    # compile supported via maybe_compile_lm_layers (optimization); fp8 via maybe_fp8_convert_lm


def get_dp_mesh(parallel_dims):
    """Extract the dp_mesh from world_mesh based on parallelism config."""
    if parallel_dims.dp_replicate_enabled:
        dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
    else:
        dp_mesh_dim_names = ("dp_shard_cp",)
    return parallel_dims.mesh[tuple(dp_mesh_dim_names)]


def build_fsdp_config(dp_mesh, config) -> dict:
    """Build the FSDP2 config dict (mp_policy + optional offload)."""
    from cosmos_rl.utils.util import str2torch_dtype
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

    param_dtype = str2torch_dtype(config.train.param_dtype)
    reduce_dtype = str2torch_dtype(config.train.fsdp_reduce_dtype)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if config.train.fsdp_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()
    return fsdp_config


def build_reshard_fn(reshard_policy: str) -> Callable[[int, int], bool]:
    """Return a callable ``(i, n) -> bool`` for reshard_after_forward."""

    def _reshard(i: int, n: int) -> bool:
        if reshard_policy == "always":
            return True
        if reshard_policy == "never":
            return False
        return i < (n - 1)

    return _reshard


def shard_visual_tower(
    model_root: torch.nn.Module,
    fsdp_config: dict,
    reshard_fn: Callable[[int, int], bool],
    model_name: str = "Model",
) -> None:
    """Apply FSDP2 to the visual tower of a VLM model."""
    from torch.distributed.fsdp import fully_shard

    visual = find_first_attr_chain(
        model_root,
        [
            ["vlm", "model", "visual"],
            ["vlm", "model", "vision_tower"],
            ["vlm", "visual"],
            ["vlm", "vision_tower"],
        ],
    )
    if visual is None:
        logger.warning(f"[{model_name}][FSDP] Could not locate visual module; skipping")
        return

    visual_blocks = getattr(visual, "blocks", None) or getattr(visual, "layers", None)
    if visual_blocks is not None:
        items = list(iter_blocks(visual_blocks))
        n_blocks = len(items)
        for idx, (_, blk) in enumerate(items):
            fully_shard(blk, **fsdp_config, reshard_after_forward=reshard_fn(idx, n_blocks))
        fully_shard(visual, **fsdp_config, reshard_after_forward=True)
        logger.info(f"[{model_name}][FSDP] Sharded visual ({n_blocks} blocks)")
    else:
        logger.warning(
            f"[{model_name}][FSDP] Visual module found but no blocks/layers; "
            "sharding as single unit"
        )
        fully_shard(visual, **fsdp_config, reshard_after_forward=True)


def shard_lm_layers(
    model_root: torch.nn.Module,
    fsdp_config: dict,
    reshard_fn: Callable[[int, int], bool],
    model_name: str = "Model",
) -> None:
    """Apply FSDP2 to language model decoder layers + embed_tokens."""
    from torch.distributed.fsdp import fully_shard

    language_model = find_first_attr_chain(
        model_root,
        [
            ["vlm", "language_model"],
            ["vlm", "model", "language_model"],
            ["vlm", "llm"],
            ["llm"],
        ],
    )
    if language_model is None:
        raise RuntimeError(f"{model_name}: could not find language model module")

    lm_core = getattr(language_model, "model", language_model)
    lm_layers = getattr(lm_core, "layers", None) or getattr(lm_core, "blocks", None)
    if lm_layers is None:
        raise RuntimeError(
            f"{model_name}: could not find decoder layers on language_model. "
            f"language_model_type={type(language_model).__name__}, "
            f"core_type={type(lm_core).__name__}"
        )

    items = list(iter_blocks(lm_layers))
    n_layers = len(items)
    for idx, (_, blk) in enumerate(items):
        fully_shard(blk, **fsdp_config, reshard_after_forward=reshard_fn(idx, n_layers))

    embed_tokens = getattr(lm_core, "embed_tokens", None)
    if embed_tokens is not None:
        fully_shard(embed_tokens, **fsdp_config, reshard_after_forward=True)
    else:
        logger.warning(f"[{model_name}][FSDP] Could not find embed_tokens; skipping")

    fully_shard(language_model, **fsdp_config, reshard_after_forward=True)
    logger.info(f"[{model_name}][FSDP] Sharded language model ({n_layers} layers)")


def maybe_fp8_convert_lm(model_root, config, parallel_dims, model_name="Model"):
    """Convert LM decoder Linear layers to torchao float8 (in-place) before FSDP sharding."""
    if config is None or not getattr(getattr(config, "train", None), "fp8", None):
        return
    if not config.train.fp8.enable_fp8:
        return
    from functools import partial
    from torchao.float8 import convert_to_float8_training
    from cosmos_rl.utils.fp8.fp8_util import FP8ModelConverter, module_filter_fn

    language_model = find_first_attr_chain(
        model_root,
        [["vlm", "language_model"], ["vlm", "model", "language_model"], ["vlm", "llm"], ["llm"]],
    )
    if language_model is None:
        logger.warning(f"[{model_name}][fp8] no language_model found; skip")
        return
    conv = FP8ModelConverter(config, parallel_dims)
    filter_fqns = [
        "lm_head", "embed_tokens", "norm", "rotary", "visual", "vision",
        "merger", "patch_embed", "expert", "action", "in_proj",
    ]
    convert_to_float8_training(
        language_model,
        config=conv.ao_float8_config,
        module_filter_fn=partial(module_filter_fn, filter_fqns=filter_fqns),
    )
    logger.info(
        f"[{model_name}][fp8] converted LM Linear -> float8 "
        f"(quant={config.train.fp8.quant_recipe}, recipe={config.train.fp8.fp8_recipe})"
    )


def maybe_compile_lm_layers(model_root, config, model_name="Model"):
    """Apply torch.compile to each LM decoder block before FSDP sharding."""
    if config is None or not getattr(getattr(config, "train", None), "compile", False):
        return
    import torch

    language_model = find_first_attr_chain(
        model_root,
        [["vlm", "language_model"], ["vlm", "model", "language_model"], ["vlm", "llm"], ["llm"]],
    )
    if language_model is None:
        logger.warning(f"[{model_name}][compile] no language_model found; skip")
        return
    lm_core = getattr(language_model, "model", language_model)
    lm_layers = getattr(lm_core, "layers", None) or getattr(lm_core, "blocks", None)
    if lm_layers is None:
        logger.warning(f"[{model_name}][compile] no decoder layers found; skip")
        return
    n = 0
    for name, blk in list(lm_layers.named_children()):
        # In-place compile: preserves module identity + param names (no _orig_mod prefix),
        # so checkpoint weight-loading by FQN still matches.
        blk.compile()
        n += 1
    logger.info(f"[{model_name}][compile] torch.compile applied to {n} LM blocks")

    # Also compile the visual tower blocks (the per-sample multi-camera ViT encode is a
    # prime suspect for the per-sample cost; compile is numerically identical).
    visual = find_first_attr_chain(
        model_root,
        [["vlm", "model", "visual"], ["vlm", "visual"], ["vlm", "model", "vision_tower"]],
    )
    if visual is not None:
        vblocks = getattr(visual, "blocks", None) or getattr(visual, "layers", None)
        if vblocks is not None:
            nv = 0
            for vname, vblk in list(vblocks.named_children()):
                vblk.compile()
                nv += 1
            logger.info(f"[{model_name}][compile] torch.compile applied to {nv} visual blocks")
