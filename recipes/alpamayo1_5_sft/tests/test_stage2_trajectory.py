# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare cached smoothing against real inference functions on CPU."""

import copy
import inspect
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from alpamayo1_5_sft.performance import stage2_trajectory as trajectory


class DisabledTrajectoryCacheTests(unittest.TestCase):
    def test_default_off_has_no_torch_import_or_model_mutation(self):
        action = object()
        model = SimpleNamespace(action_space=action)
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"torch": None}):
            result = trajectory.apply_trajectory_factor_cache(model)
            self.assertEqual(result, {
                "requested": False, "enabled": False, "runtime_enabled": False,
                "patched": False, "patched_instance_count": 0, "instances": [],
            })
            self.assertIs(model.action_space, action)


class RealTrajectoryCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from alpamayo_r1.action_space.discrete_action_space import DiscreteTrajectoryTokenizer
            from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
        except ImportError as exc:
            raise unittest.SkipTest(f"Real torch/inference dependencies unavailable: {exc}") from exc
        cls.torch = torch
        cls.action_type = UnicycleAccelCurvatureActionSpace
        cls.tokenizer_type = DiscreteTrajectoryTokenizer

    def setUp(self):
        switches = patch.dict(os.environ, {"APPLY_TRAJECTORY_FACTOR_CACHE": "1"}, clear=True)
        switches.start()
        self.addCleanup(switches.stop)

    def _action(self, length=64):
        return self.action_type(n_waypoints=length)

    def _tokenizer(self, length):
        return self.tokenizer_type(
            action_space_cfg={
                "_target_": "alpamayo_r1.action_space.unicycle_accel_curvature.UnicycleAccelCurvatureActionSpace",
                "n_waypoints": length,
            },
            dims_min=[-10.0, -0.5], dims_max=[10.0, 0.5], num_bins=384,
        )

    def _inputs(self, length=64, history=16, batch=1, speed=3.0, yaw_rate=0.0, dtype=None):
        torch = self.torch
        dtype = dtype or torch.float32

        def sample(times):
            yaw = yaw_rate * times
            xyz = torch.zeros(len(times), 3, dtype=dtype)
            if yaw_rate:
                xyz[:, 0] = speed / yaw_rate * yaw.sin()
                xyz[:, 1] = speed / yaw_rate * (1 - yaw.cos())
            else:
                xyz[:, 0] = speed * times
            rotations = torch.eye(3, dtype=dtype).repeat(len(times), 1, 1)
            rotations[:, 0, 0] = rotations[:, 1, 1] = yaw.cos()
            rotations[:, 1, 0] = yaw.sin()
            rotations[:, 0, 1] = -yaw.sin()
            return xyz.repeat(batch, 1, 1), rotations.repeat(batch, 1, 1, 1)

        hist_xyz, hist_rot = sample(torch.arange(1 - history, 1, dtype=dtype) * 0.1)
        fut_xyz, fut_rot = sample(torch.arange(1, length + 1, dtype=dtype) * 0.1)
        return hist_xyz, hist_rot, fut_xyz, fut_rot

    def _apply(self, model):
        with patch.dict(os.environ, {"APPLY_TRAJECTORY_FACTOR_CACHE": "1"}, clear=True):
            return trajectory.apply_trajectory_factor_cache(model)

    def test_targets_match_original_across_lengths_heading_wrap_and_low_speed(self):
        torch = self.torch
        for length in (16, 64):
            original = self._action(length)
            cached = self._action(length)
            model = SimpleNamespace(action_space=cached)
            old_signature = inspect.signature(cached.traj_to_action)
            old_keys = list(cached.state_dict())
            self._apply(model)
            self.assertEqual(inspect.signature(cached.traj_to_action), old_signature)
            for batch in (1, 2):
                for speed, rate in ((0.0, 0.0), (0.59, 0.0), (0.61, 0.0), (3.0, 0.8)):
                    inputs = self._inputs(length, batch=batch, speed=speed, yaw_rate=rate)
                    expected = original.traj_to_action(*inputs)
                    for _ in range(2):
                        actual = cached.traj_to_action(*inputs)
                        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                        self.assertTrue(bool(torch.isfinite(actual).all()))
            self.assertEqual(list(cached.state_dict()), old_keys)
            stats = trajectory.trajectory_factor_cache_stats(model)["instances"][0]
            self.assertGreater(stats["hits"]["theta"], 0)
            self.assertGreater(stats["hits"]["accel"], 0)
            self.assertEqual(stats["build_failures"], {"theta": 0, "accel": 0})

    def test_real_history_and_future_tokens_match_with_shared_alias_detection(self):
        torch = self.torch
        history = self._tokenizer(16)
        future = self._tokenizer(64)
        original_history = copy.deepcopy(history)
        original_future = copy.deepcopy(future)
        model = SimpleNamespace(action_space=future.action_space, hist_traj_tokenizer=history, traj_tokenizer=future)
        report = self._apply(model)
        self.assertEqual(report["patched_instance_count"], 2)
        for speed, rate in ((0.0, 0.0), (0.59, 0.0), (0.61, 0.0), (3.0, 0.8)):
            hist_xyz, hist_rot, fut_xyz, fut_rot = self._inputs(speed=speed, yaw_rate=rate)
            args = (hist_xyz[:, :1], hist_rot[:, :1], hist_xyz, hist_rot)
            expected_history = original_history.encode(*args)
            expected_future = original_future.encode(hist_xyz, hist_rot, fut_xyz, fut_rot)
            for _ in range(2):
                torch.testing.assert_close(history.encode(*args), expected_history, rtol=0, atol=0)
                torch.testing.assert_close(future.encode(hist_xyz, hist_rot, fut_xyz, fut_rot), expected_future, rtol=0, atol=0)
        again = self._apply(model)
        self.assertEqual(again["patched_instance_count"], 2)
        aliases = [entry["aliases"] for entry in again["instances"]]
        self.assertIn(["action_space", "traj_tokenizer.action_space"], aliases)

    def test_configuration_shape_and_dtype_changes_miss_and_lru_is_bounded(self):
        torch = self.torch
        original = self._action()
        cached = self._action()
        model = SimpleNamespace(action_space=cached)
        self._apply(model)
        inputs = self._inputs()
        cached.traj_to_action(*inputs)
        for index in range(1, 7):
            original.theta_lambda = cached.theta_lambda = index * 1e-5
            original.a_lambda = cached.a_lambda = (index + 1) * 1e-4
            torch.testing.assert_close(cached.traj_to_action(*inputs), original.traj_to_action(*inputs), rtol=1e-6, atol=1e-7)
        for batch, dtype in ((2, torch.float32), (1, torch.float64)):
            varied = self._inputs(batch=batch, dtype=dtype)
            torch.testing.assert_close(cached.traj_to_action(*varied), original.traj_to_action(*varied), rtol=1e-6, atol=1e-7)
        stats = trajectory.trajectory_factor_cache_stats(model)["instances"][0]
        self.assertGreater(stats["misses"]["theta"], 8)
        self.assertGreater(stats["misses"]["accel"], 8)
        self.assertEqual(stats["entries"], stats["capacity"])
        self.assertGreater(stats["evictions"], 0)

    def test_factor_failure_falls_back_to_original_solvers_without_global_patch(self):
        torch = self.torch
        original = self._action()
        cached = self._action()
        model = SimpleNamespace(action_space=cached)
        self._apply(model)
        inputs = self._inputs(speed=0.61, yaw_rate=0.8)
        expected = original.traj_to_action(*inputs)
        # Fault injection targets our builder only; torch.linalg and every
        # original inference solver remain real and unmodified.
        with patch.object(trajectory, "_build_fixed_factor", side_effect=RuntimeError("test factor failure")):
            torch.testing.assert_close(cached.traj_to_action(*inputs), expected, rtol=1e-6, atol=1e-7)
        stats = trajectory.trajectory_factor_cache_stats(model)["instances"][0]
        self.assertEqual(stats["entries"], 0)
        self.assertEqual(stats["build_failures"], {"theta": 1, "accel": 1})
        self.assertEqual(stats["fallbacks"], {"theta": 1, "accel": 1})
        self.assertEqual(stats["fallback_reasons"], {
            "theta": {"checked_factor_build_failed": 1},
            "accel": {"checked_factor_build_failed": 1},
        })

    def test_real_hf_configuration_with_hydra_listconfig_bounds_activates(self):
        from hydra.utils import instantiate

        torch = self.torch
        # Actual action_space_cfg from original-alpamayo15/config.json. The
        # model constructor uses Hydra's default conversion, retaining bounds
        # as ListConfig rather than converting them to Python list/tuple.
        config = json.loads('''{
          "_target_": "alpamayo_r1.action_space.UnicycleAccelCurvatureActionSpace",
          "a_lambda": 0.0001, "a_ridge": 0.0001,
          "accel_bounds": [-9.8, 9.8],
          "accel_mean": 0.02902694707164455, "accel_std": 0.6810426736454882,
          "curvature_bounds": [-0.33, 0.33],
          "curvature_mean": 0.0002692167976330542,
          "curvature_std": 0.026148280660833106,
          "dt": 0.1, "kappa_lambda": 0.0001, "kappa_ridge": 0.0001,
          "n_waypoints": 64, "theta_lambda": 0.000001,
          "theta_ridge": 0.00000001, "v_lambda": 0.000001, "v_ridge": 0.0001
        }''')
        original = instantiate(config).to(dtype=torch.bfloat16)
        cached = instantiate(config).to(dtype=torch.bfloat16)
        self.assertEqual(type(cached.accel_bounds).__name__, "ListConfig")
        self.assertEqual(type(cached.curvature_bounds).__name__, "ListConfig")
        model = SimpleNamespace(action_space=cached)
        self._apply(model)
        inputs = self._inputs(speed=0.61, yaw_rate=0.8)
        expected = original.traj_to_action(*inputs)
        for _ in range(2):
            torch.testing.assert_close(cached.traj_to_action(*inputs), expected, rtol=1e-6, atol=1e-7)
        stats = trajectory.trajectory_factor_cache_stats(model)["instances"][0]
        self.assertEqual(stats["entries"], 2)
        self.assertEqual(stats["misses"], {"theta": 1, "accel": 1})
        self.assertEqual(stats["hits"], {"theta": 1, "accel": 1})
        self.assertEqual(stats["fallback_reasons"], {"theta": {}, "accel": {}})

    def test_eligibility_diagnostics_and_role_specific_configuration(self):
        torch = self.torch
        cached = self._action()
        values = torch.ones(1, 64)
        # An unrelated dynamic solver setting cannot invalidate/reject a fixed
        # theta factor. Its original dynamic solver still handles that setting.
        before = trajectory._cache_key(cached, "theta", values, torch)
        cached.v_lambda *= 10
        self.assertTrue(trajectory._eligible(cached, "theta", values, torch))
        self.assertEqual(before, trajectory._cache_key(cached, "theta", values, torch))
        cached.theta_lambda = torch.tensor(1e-6)
        reason = trajectory._eligibility_failure(cached, "theta", values, torch)
        self.assertIn("unsupported_config_type:theta_lambda:torch.Tensor", reason)
        model = SimpleNamespace(action_space=cached)
        self._apply(model)
        cached.traj_to_action(*self._inputs())
        stats = trajectory.trajectory_factor_cache_stats(model)["instances"][0]
        self.assertEqual(stats["fallback_reasons"]["theta"], {reason: 1})
        self.assertEqual(stats["last_input"]["theta"]["eligibility_failure"], reason)

    def test_output_buffer_broadcasts_preserve_hits_and_live_normalization(self):
        torch = self.torch
        original, cached = self._action(), self._action()
        model = SimpleNamespace(action_space=cached)
        self._apply(model)
        inputs = self._inputs(speed=0.61, yaw_rate=0.8)
        cached.traj_to_action(*inputs)
        for step in range(3):
            # DDP broadcasts mutate buffer versions on every forward. Also
            # exercise changed output scaling: it must read the live buffers.
            for name in ("accel_mean", "accel_std", "curvature_mean", "curvature_std"):
                target = getattr(cached, name)
                value = getattr(original, name) + (0.01 if step == 2 else 0.0)
                getattr(original, name).copy_(value)
                target.copy_(value)
            torch.testing.assert_close(cached.traj_to_action(*inputs), original.traj_to_action(*inputs), rtol=1e-6, atol=1e-7)
        stats = trajectory.trajectory_factor_cache_stats(model)["instances"][0]
        self.assertEqual(stats["misses"], {"theta": 1, "accel": 1})
        self.assertEqual(stats["hits"], {"theta": 3, "accel": 3})

    def test_amp_context_is_part_of_key_even_with_same_float32_input(self):
        torch = self.torch
        action = self._action()
        values = torch.ones(1, 64, dtype=torch.float32)
        base_key = trajectory._cache_key(action, "theta", values, torch)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            amp_key = trajectory._cache_key(action, "theta", values, torch)
        self.assertNotEqual(base_key, amp_key)
        self.assertEqual(base_key[-1][0], ("cpu", False, str(torch.get_autocast_dtype("cpu"))))
        self.assertTrue(amp_key[-1][0][1])
        self.assertEqual(base_key, trajectory._cache_key(action, "theta", values, torch))

    def test_runtime_disable_delegates_without_changing_counters(self):
        torch = self.torch
        cached = self._action()
        original = self._action()
        model = SimpleNamespace(action_space=cached)
        self._apply(model)
        inputs = self._inputs(yaw_rate=0.8)
        cached.traj_to_action(*inputs)
        before = trajectory.trajectory_factor_cache_stats(model)["instances"]
        with patch.dict(os.environ, {"APPLY_TRAJECTORY_FACTOR_CACHE": "0"}):
            torch.testing.assert_close(cached.traj_to_action(*inputs), original.traj_to_action(*inputs), rtol=0, atol=0)
            after = trajectory.trajectory_factor_cache_stats(model)
            self.assertFalse(after["runtime_enabled"])
            self.assertFalse(after["enabled"])
            self.assertTrue(after["patched"])
            self.assertEqual(before, after["instances"])

    def test_optional_states_initial_velocity_and_checkpoint_buffers_are_preserved(self):
        torch = self.torch
        original = self._action()
        cached = self._action()
        model = SimpleNamespace(action_space=cached)
        buffers = {name: value for name, value in cached.named_buffers()}
        self._apply(model)
        inputs = self._inputs(speed=3.0, yaw_rate=0.8)
        initial = {"v": torch.tensor([2.8])}
        actual = cached.traj_to_action(*inputs, t0_states=initial, output_all_states=True)
        expected = original.traj_to_action(*inputs, t0_states=initial, output_all_states=True)
        for value, ref in zip(actual, expected):
            torch.testing.assert_close(value, ref, rtol=1e-6, atol=1e-7)
        for name, value in cached.named_buffers():
            self.assertIs(value, buffers[name])
        restored = self._action()
        restored.load_state_dict(cached.state_dict(), strict=True)
        torch.testing.assert_close(restored.traj_to_action(*inputs), original.traj_to_action(*inputs), rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
