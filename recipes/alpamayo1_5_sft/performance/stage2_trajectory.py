# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in bounded caches for fixed Stage-2 trajectory smoothing systems.

Only theta smoothing and acceleration smoothing have fixed left-hand sides.
Velocity and curvature systems still use the original inference functions. The
implementation follows NVlabs/alpamayo f3198c2; it never edits installed files or
patches global torch/inference functions. Importing this module needs no torch.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import wraps
from types import MethodType
from typing import Any
import inspect
import math

from alpamayo1_5_sft.performance.stage2_optimizations import stage2_flag


_STATE = "_stage2_trajectory_factor_cache"
_CAPACITY = 8
_FACTOR_ATTRIBUTES = {
    "theta": ("n_waypoints", "dt", "theta_lambda", "theta_ridge"),
    "accel": ("n_waypoints", "dt", "a_lambda", "a_ridge"),
}


def _action_spaces(model: Any) -> list[tuple[list[str], Any]]:
    unique: dict[int, tuple[list[str], Any]] = {}
    candidates = [("action_space", getattr(model, "action_space", None))]
    for name in ("hist_traj_tokenizer", "traj_tokenizer"):
        tokenizer = getattr(model, name, None)
        candidates.append((name + ".action_space", getattr(tokenizer, "action_space", None)))
    for name, action_space in candidates:
        if action_space is None:
            continue
        identity = id(action_space)
        if identity not in unique:
            unique[identity] = ([name], action_space)
        else:
            unique[identity][0].append(name)
    return list(unique.values())


def _static_configuration(action_space: Any, role: str) -> tuple[Any, ...]:
    # Bounds/velocity/curvature settings do not enter either cached factor.
    # Hydra deliberately retains bounds as ListConfig in real HF checkpoints;
    # neither that container nor unrelated configuration should reject a valid
    # theta/acceleration system. Affine metadata is conservative invalidation.
    fields = []
    for name in _FACTOR_ATTRIBUTES[role]:
        value = getattr(action_space, name)
        fields.append((name, value))
    for name in ("accel_mean", "accel_std", "curvature_mean", "curvature_std"):
        value = getattr(action_space, name)
        try:
            version = value._version
        except RuntimeError:  # Inference-mode tensors have no version counter.
            version = None
        fields.append((name, id(value), version, str(value.dtype), str(value.device)))
    return tuple(fields)


def _cache_key(action_space: Any, role: str, y: Any, torch: Any) -> tuple[Any, ...]:
    smooth = (None, None, 1.0) if role == "theta" else (None, 1.0, None)
    autocast = tuple(
        (device, torch.is_autocast_enabled(device), str(torch.get_autocast_dtype(device)))
        for device in ("cpu", "cuda")
    )
    return (
        role, _static_configuration(action_space, role), y.shape[-1], tuple(y.shape[:-1]),
        str(y.dtype), y.device.type, y.device.index,
        "unit_data_weights", "identity_data_operator", smooth, autocast,
    )


def _eligibility_failure(action_space: Any, role: str, y: Any, torch: Any) -> str | None:
    lam = action_space.theta_lambda if role == "theta" else action_space.a_lambda
    ridge = action_space.theta_ridge if role == "theta" else action_space.a_ridge
    # The supported constructor takes Python scalars/tuples. Do not inspect a
    # Tensor-valued custom configuration with float()/item(): that could both
    # synchronize a GPU and invalidate a factor without changing its hash.
    for name in _FACTOR_ATTRIBUTES[role]:
        field = getattr(action_space, name)
        if type(field) not in (int, float):
            return f"unsupported_config_type:{name}:{type(field).__module__}.{type(field).__name__}"
        if not math.isfinite(field):
            return f"non_finite_config:{name}"
    if action_space.dt <= 0:
        return "non_positive_dt"
    if lam < 0:
        return "negative_smoothing_lambda"
    if ridge < 0:
        return "negative_ridge"
    if not y.ndim or y.shape[-1] <= 0:
        return "empty_input_sequence"
    if y.dtype not in (torch.float32, torch.float64):
        return f"unsupported_input_dtype:{y.dtype}"
    return None


def _eligible(action_space: Any, role: str, y: Any, torch: Any) -> bool:
    return _eligibility_failure(action_space, role, y, torch) is None


def _fallback(state: dict[str, Any], role: str, reason: str) -> None:
    state["fallbacks"][role] += 1
    reasons = state["fallback_reasons"][role]
    reasons[reason] = reasons.get(reason, 0) + 1


def _build_fixed_factor(action_space: Any, role: str, y: Any, torch: Any, utils: Any, einops: Any) -> dict[str, Any]:
    """Build the original constant matrix and perform its original checked solve."""
    *lead, n = y.shape
    device, dtype = y.device, y.dtype
    weights = torch.ones_like(y)
    if role == "theta":
        data_operator = torch.eye(n, dtype=dtype, device=device).expand(*lead, n, n)
        lam, ridge = action_space.theta_lambda, action_space.theta_ridge
    else:
        # The original _v_to_a passes s=ones_like(dv), w_data=None.
        data_operator = torch.diag_embed(torch.ones_like(y))
        lam, ridge = action_space.a_lambda, action_space.a_ridge
    weighted_operator = data_operator * weights.unsqueeze(-1)
    ata = einops.einsum(weighted_operator, data_operator, "... i j, ... i k -> ... j k")
    if role == "theta":
        dtd = utils.construct_DTD(
            n + 1, lead, device=device, dtype=dtype,
            w_smooth1=None, w_smooth2=None, w_smooth3=1.0,
            lam=lam, dt=action_space.dt,
        )
        smoothing = dtd[..., 1:, 1:]
        boundary = dtd[..., 1:, 0].clone()
    else:
        smoothing = utils.construct_DTD(
            n, lead, device=device, dtype=dtype,
            w_smooth1=None, w_smooth2=1.0, w_smooth3=None,
            lam=lam, dt=action_space.dt,
        )
        boundary = None
    ridge_term = ridge * torch.eye(n, dtype=dtype, device=device).expand(*lead, n, n)
    lhs = ata + smoothing + ridge_term
    # Do not replace this with unchecked cholesky_ex: the existing failure
    # behavior and the acceleration solver's adaptive-ridge fallback matter.
    factor = torch.linalg.cholesky(lhs)
    return {"factor": factor, "weighted_operator": weighted_operator, "boundary": boundary}


def _factor(action_space: Any, role: str, y: Any, state: dict[str, Any], torch: Any, utils: Any, einops: Any) -> dict[str, Any] | None:
    reason = _eligibility_failure(action_space, role, y, torch)
    state["last_input"][role] = {
        "shape": list(y.shape), "dtype": str(y.dtype), "device": str(y.device),
        "eligibility_failure": reason,
    }
    if reason is not None:
        _fallback(state, role, reason)
        return None
    key = _cache_key(action_space, role, y, torch)
    cache = state["cache"]
    if key in cache:
        state["hits"][role] += 1
        cache.move_to_end(key)
        return cache[key]
    state["misses"][role] += 1
    try:
        payload = _build_fixed_factor(action_space, role, y, torch, utils, einops)
    except RuntimeError as exc:
        # No failed factor is cached. The caller invokes its unchanged original
        # solver, which either raises normally or retries with an increased ridge.
        state["build_failures"][role] += 1
        _fallback(state, role, "checked_factor_build_failed")
        state["last_build_error"][role] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        return None
    cache[key] = payload
    if len(cache) > state["capacity"]:
        cache.popitem(last=False)
        state["evictions"] += 1
    return payload


def _patch_instance(action_space: Any, torch: Any, utils: Any, einops: Any) -> None:
    original = action_space.traj_to_action
    original_acceleration = action_space._v_to_a
    state: dict[str, Any] = {
        "capacity": _CAPACITY, "cache": OrderedDict(),
        "hits": {"theta": 0, "accel": 0}, "misses": {"theta": 0, "accel": 0},
        "build_failures": {"theta": 0, "accel": 0},
        "fallbacks": {"theta": 0, "accel": 0}, "evictions": 0,
        "fallback_reasons": {"theta": {}, "accel": {}},
        "last_input": {}, "last_build_error": {},
    }

    def smooth_theta(rotations):
        theta = utils.unwrap_angle(utils.so3_to_yaw_torch(rotations))
        payload = _factor(action_space, "theta", theta, state, torch, utils, einops)
        if payload is None:
            return utils.theta_smooth(
                traj_future_rot=rotations, dt=action_space.dt,
                theta_lambda=action_space.theta_lambda, theta_ridge=action_space.theta_ridge,
            )
        initial = torch.zeros_like(theta[..., 0])
        rhs = einops.einsum(payload["weighted_operator"], theta, "... i j, ... i -> ... j")
        rhs -= payload["boundary"] * initial.unsqueeze(-1)
        values = torch.cholesky_solve(rhs.unsqueeze(-1), payload["factor"]).squeeze(-1)
        return torch.cat([initial.unsqueeze(-1), values], dim=-1)

    def smooth_acceleration(velocity):
        dv = (velocity[..., 1:] - velocity[..., :-1]) / action_space.dt
        payload = _factor(action_space, "accel", dv, state, torch, utils, einops)
        if payload is None:
            return original_acceleration(velocity)
        rhs = einops.einsum(payload["weighted_operator"], dv, "... i j, ... i -> ... j")
        # Preserve solve_xs_eq_y's explicit RHS cast (relevant to CPU AMP too).
        if rhs.dtype != payload["factor"].dtype:
            rhs = rhs.to(payload["factor"].dtype)
        return torch.cholesky_solve(rhs.unsqueeze(-1), payload["factor"]).squeeze(-1)

    @torch.no_grad()
    @torch._dynamo.disable()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def cached_traj_to_action(
        self, traj_history_xyz, traj_history_rot, traj_future_xyz, traj_future_rot,
        t0_states=None, output_all_states=False,
    ):
        # Original f3198c2 algorithm and call order. Only the two fixed smoothing
        # systems above use cached factors. All data-dependent systems stay in
        # the installed inference implementation, including its failure handling.
        if traj_future_xyz.shape[-2] != self.n_waypoints:
            return original(
                traj_history_xyz, traj_history_rot, traj_future_xyz, traj_future_rot,
                t0_states=t0_states, output_all_states=output_all_states,
            )
        if t0_states is None:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)
        full_xy = torch.cat([traj_history_xyz[..., -1:, :], traj_future_xyz], dim=-2)[..., :2]
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
        theta = smooth_theta(traj_future_rot)
        velocity = utils.dxy_theta_to_v(
            dxy=dxy, theta=theta, v0=t0_states["v"], dt=self.dt,
            v_lambda=self.v_lambda, v_ridge=self.v_ridge,
        )
        accel = smooth_acceleration(velocity)
        kappa = self._theta_v_a_to_kappa(theta, velocity, accel)
        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = (accel - accel_mean) / accel_std
        kappa = (kappa - kappa_mean) / kappa_std
        if not output_all_states:
            return torch.stack([accel, kappa], dim=-1)
        return torch.stack([accel, kappa], dim=-1), torch.stack(
            [velocity[:, :-1], accel, theta[:, :-1]], dim=-1,
        )

    @wraps(original.__func__)
    def traj_to_action(
        self, traj_history_xyz, traj_history_rot, traj_future_xyz, traj_future_rot,
        t0_states=None, output_all_states=False,
    ):
        # Dispatch outside every added AMP/no_grad context: flag=0 is exactly
        # the original bound call, including its own original decorators. This
        # permits sequential correctness comparisons on the same model.
        if not stage2_flag("APPLY_TRAJECTORY_FACTOR_CACHE"):
            return original(
                traj_history_xyz, traj_history_rot, traj_future_xyz, traj_future_rot,
                t0_states=t0_states, output_all_states=output_all_states,
            )
        return cached_traj_to_action(
            self, traj_history_xyz, traj_history_rot, traj_future_xyz, traj_future_rot,
            t0_states=t0_states, output_all_states=output_all_states,
        )

    # Plain private Python state is deliberately not registered as parameters,
    # child modules or buffers, so checkpoint keys and parameter identity stay put.
    setattr(action_space, _STATE, state)
    action_space.traj_to_action = MethodType(traj_to_action, action_space)


def trajectory_factor_cache_stats(model: Any) -> dict[str, Any]:
    """Return JSON-compatible per-instance counters without importing torch."""
    instances = []
    for aliases, action_space in _action_spaces(model):
        state = getattr(action_space, _STATE, None)
        if state is None:
            continue
        tensor_bytes = sum(
            value.numel() * value.element_size()
            for payload in state["cache"].values()
            for value in payload.values() if value is not None
        )
        instances.append({
            "aliases": aliases, "class": type(action_space).__name__,
            "capacity": state["capacity"], "entries": len(state["cache"]),
            "cached_tensor_bytes": tensor_bytes,
            "hits": dict(state["hits"]), "misses": dict(state["misses"]),
            "build_failures": dict(state["build_failures"]),
            "fallbacks": dict(state["fallbacks"]), "evictions": state["evictions"],
            "fallback_reasons": {role: dict(reasons) for role, reasons in state["fallback_reasons"].items()},
            "last_input": {role: dict(value) for role, value in state["last_input"].items()},
            "last_build_error": {role: dict(value) for role, value in state["last_build_error"].items()},
        })
    runtime_enabled = stage2_flag("APPLY_TRAJECTORY_FACTOR_CACHE")
    return {
        "enabled": bool(instances) and runtime_enabled,
        "runtime_enabled": runtime_enabled, "patched": bool(instances),
        "patched_instance_count": len(instances), "instances": instances,
    }


def apply_trajectory_factor_cache(model: Any) -> dict[str, Any]:
    """Enable fixed theta/accel factor caches only with APPLY_TRAJECTORY_FACTOR_CACHE=1.

    Call after loading the model and before Trainer/DDP. Only the exact supported
    action-space class is patched, independently on the continuous target and
    history/future tokenizer instances. Unsupported subclasses are reported, not
    rewritten. Factors are lazy, bounded and invalidated by their complete key.
    """
    requested = stage2_flag("APPLY_TRAJECTORY_FACTOR_CACHE")
    if not requested:
        return {"requested": False, **trajectory_factor_cache_stats(model)}
    try:
        import torch
        import einops
        from alpamayo_r1.action_space import utils
        from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
    except ImportError as exc:
        raise RuntimeError(
            "APPLY_TRAJECTORY_FACTOR_CACHE=1 requires torch and the compatible "
            "alpamayo-r1 UnicycleAccelCurvatureActionSpace (f3198c2 implementation)"
        ) from exc
    skipped = []
    for aliases, action_space in _action_spaces(model):
        if type(action_space) is not UnicycleAccelCurvatureActionSpace:
            skipped.append({"aliases": aliases, "class": type(action_space).__name__, "reason": "unsupported_exact_class"})
            continue
        if getattr(action_space, _STATE, None) is None:
            expected = (
                "self", "traj_history_xyz", "traj_history_rot", "traj_future_xyz",
                "traj_future_rot", "t0_states", "output_all_states",
            )
            actual = tuple(inspect.signature(type(action_space).traj_to_action).parameters)
            if actual != expected:
                raise TypeError(f"Unsupported trajectory conversion API: {actual!r}")
            if action_space.traj_to_action.__func__ is not type(action_space).traj_to_action:
                raise TypeError("Refusing to replace another instance-specific traj_to_action patch")
            if action_space._v_to_a.__func__ is not type(action_space)._v_to_a:
                raise TypeError("Refusing to replace another instance-specific acceleration solver")
            _patch_instance(action_space, torch, utils, einops)
    result = {"requested": True, **trajectory_factor_cache_stats(model), "skipped_instances": skipped}
    if not result["enabled"]:
        raise TypeError("No supported UnicycleAccelCurvatureActionSpace instance was found")
    return result
