# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary tests and an optional real tiny-Qwen3-VL CPU prefix-KV check.

The tiny model check is deliberately limited to a text-only, zero-dropout CPU
case. It does not establish equivalence for real PAI visual inputs or bf16 GPUs.
"""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alpamayo1_5_sft.performance.stage2_prefix import maybe_trim_future_prefix


class PrefixSwitchTests(unittest.TestCase):
    def test_disabled_preserves_identity_without_touching_model_or_torch(self):
        ids = SimpleNamespace(shape=(1, 6))
        kwargs = {"attention_mask": object()}
        with patch.dict(os.environ, {}, clear=True):
            result_ids, result_kwargs, report = maybe_trim_future_prefix(object(), ids, kwargs)
        self.assertIs(result_ids, ids)
        self.assertIs(result_kwargs, kwargs)
        self.assertFalse(report["applied"])
        self.assertEqual(report["tokens_before"], 6)
        self.assertEqual(report["tokens_after"], 6)
        self.assertEqual(report["fallback_reason"], "disabled")

    def test_joint_training_and_missing_skip_head_fall_back_before_dependencies(self):
        ids, kwargs = object(), {}
        env = {"APPLY_VLM_PREFIX_ONLY": "1", "APPLY_VLM_SKIP_HEAD": "1"}
        with patch.dict(os.environ, env, clear=True):
            outputs = maybe_trim_future_prefix(SimpleNamespace(cotrain_vlm=True), ids, kwargs)
        self.assertIs(outputs[0], ids)
        self.assertIs(outputs[1], kwargs)
        self.assertEqual(outputs[2]["fallback_reason"], "joint_vlm_training")
        env["APPLY_VLM_SKIP_HEAD"] = "0"
        with patch.dict(os.environ, env, clear=True):
            outputs = maybe_trim_future_prefix(SimpleNamespace(cotrain_vlm=False), ids, kwargs)
        self.assertEqual(outputs[2]["fallback_reason"], "skip_head_required")

    def test_invalid_switch_is_not_accepted(self):
        with patch.dict(os.environ, {"APPLY_VLM_PREFIX_ONLY": "tru"}, clear=True):
            with self.assertRaisesRegex(ValueError, "APPLY_VLM_PREFIX_ONLY"):
                maybe_trim_future_prefix(object(), object(), {})


class RealQwenPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise unittest.SkipTest(f"Real CPU Qwen3-VL dependencies unavailable: {exc}") from exc
        cls.torch = torch
        cls.config_type = Qwen3VLConfig
        cls.model_type = Qwen3VLForConditionalGeneration

    def setUp(self):
        config = self.config_type(
            text_config={
                "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 1, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 8, "max_position_embeddings": 64,
                "attention_dropout": 0.0,
                "rope_scaling": {"mrope_section": [1, 1, 2], "rope_type": "default"},
            },
            vision_config={
                "depth": 1, "hidden_size": 16, "intermediate_size": 32,
                "num_heads": 4, "in_channels": 3, "patch_size": 2,
                "spatial_merge_size": 2, "temporal_patch_size": 2,
                "out_hidden_size": 32, "deepstack_visual_indexes": [],
            },
            image_token_id=60, video_token_id=61, vision_start_token_id=62,
            vision_end_token_id=63,
        )
        vlm = self.model_type(config).train()
        vlm.requires_grad_(False)
        self.model = SimpleNamespace(
            cotrain_vlm=False, vlm=vlm,
            config=SimpleNamespace(traj_token_ids={"future_start": 7}),
        )
        self.env = {"APPLY_VLM_PREFIX_ONLY": "1", "APPLY_VLM_SKIP_HEAD": "1"}
        self.ids = self.torch.tensor([[3, 4, 7, 5, 6]])
        self.kwargs = {"attention_mask": self.torch.ones_like(self.ids)}

    def call(self, ids=None, kwargs=None):
        with patch.dict(os.environ, self.env, clear=True):
            return maybe_trim_future_prefix(
                self.model, self.ids if ids is None else ids,
                self.kwargs if kwargs is None else kwargs,
            )

    def assert_fallback(self, ids, kwargs, reason):
        result_ids, result_kwargs, report = self.call(ids, kwargs)
        self.assertIs(result_ids, ids)
        self.assertIs(result_kwargs, kwargs)
        self.assertFalse(report["applied"])
        self.assertEqual(report["fallback_reason"], reason)

    def test_success_keeps_boundary_and_never_changes_inputs_or_visual_objects(self):
        torch = self.torch
        pixels, grid = torch.randn(4, 8), torch.tensor([[1, 2, 2]])
        kwargs = {**self.kwargs, "pixel_values": pixels, "image_grid_thw": grid}
        original_ids, original_mask = self.ids.clone(), kwargs["attention_mask"].clone()
        ids, result, report = self.call(kwargs=kwargs)
        torch.testing.assert_close(ids, torch.tensor([[3, 4, 7]]))
        self.assertIsNot(result, kwargs)
        self.assertIs(result["pixel_values"], pixels)
        self.assertIs(result["image_grid_thw"], grid)
        torch.testing.assert_close(self.ids, original_ids)
        torch.testing.assert_close(kwargs["attention_mask"], original_mask)
        self.assertEqual(result["attention_mask"].shape, (1, 3))
        self.assertEqual((report["tokens_before"], report["tokens_after"]), (5, 3))

    def test_requires_one_boundary_no_padding_and_batch_one(self):
        torch = self.torch
        for values in ([3, 4, 5], [7, 4, 7, 5]):
            ids = torch.tensor([values])
            self.assert_fallback(
                ids, {"attention_mask": torch.ones_like(ids)},
                "requires_exactly_one_future_start",
            )
        self.assert_fallback(
            self.ids, {"attention_mask": torch.tensor([[1, 1, 1, 1, 0]])},
            "padding_or_nonfull_attention_mask",
        )
        ids = self.ids.repeat(2, 1)
        self.assert_fallback(
            ids, {"attention_mask": torch.ones_like(ids)}, "requires_nonempty_batch_one",
        )
        self.assert_fallback(self.ids, {}, "requires_full_attention_mask")

    def test_visual_tokens_after_boundary_always_fall_back(self):
        torch = self.torch
        for token in (60, 61, 62, 63):
            ids = torch.tensor([[3, 7, token, 4]])
            self.assert_fallback(
                ids, {"attention_mask": torch.ones_like(ids)},
                "visual_tokens_after_future_start",
            )

    def test_boundary_at_first_or_last_token_is_handled_inclusively(self):
        torch = self.torch
        ids = torch.tensor([[7, 3, 4]])
        result, kwargs, report = self.call(ids, {"attention_mask": torch.ones_like(ids)})
        torch.testing.assert_close(result, torch.tensor([[7]]))
        self.assertEqual(kwargs["attention_mask"].shape[-1], 1)
        self.assertTrue(report["applied"])
        ids = torch.tensor([[3, 4, 7]])
        self.assert_fallback(
            ids, {"attention_mask": torch.ones_like(ids)}, "no_suffix_to_trim",
        )

    def test_explicit_positions_past_unknown_inputs_and_nondefault_rope_fall_back(self):
        for key in ("position_ids", "cache_position", "rope_deltas"):
            self.assert_fallback(
                self.ids, {**self.kwargs, key: None}, f"explicit_{key}",
            )
        self.assert_fallback(
            self.ids, {**self.kwargs, "past_key_values": object()}, "existing_past_key_values",
        )
        self.assert_fallback(
            self.ids, {**self.kwargs, "token_type_ids": self.ids},
            "unsupported_kwargs:token_type_ids",
        )
        with patch.object(
            self.model.vlm.config.text_config, "rope_scaling", {"rope_type": "dynamic"},
        ):
            self.assert_fallback(self.ids, self.kwargs, "nondefault_rope")

    def test_unfrozen_weights_or_nonzero_attention_dropout_fall_back(self):
        parameter = next(self.model.vlm.parameters())
        parameter.requires_grad_(True)
        self.assert_fallback(self.ids, self.kwargs, "unfrozen_vlm")
        parameter.requires_grad_(False)
        with patch.object(self.model.vlm.config.text_config, "attention_dropout", 0.1):
            self.assert_fallback(self.ids, self.kwargs, "nonzero_attention_dropout")

    def test_real_tiny_qwen_cpu_prefix_kv_and_rope_delta_match_full_text(self):
        torch = self.torch
        prefix_ids, prefix_kwargs, report = self.call()
        self.assertTrue(report["applied"])
        with torch.no_grad():
            full = self.model.vlm(
                input_ids=self.ids, **self.kwargs, use_cache=True, logits_to_keep=1,
            )
            prefix = self.model.vlm(
                input_ids=prefix_ids, **prefix_kwargs, use_cache=True, logits_to_keep=1,
            )
        self.assertEqual(prefix.past_key_values.get_seq_length(), 3)
        for full_layer, prefix_layer in zip(
            full.past_key_values.layers, prefix.past_key_values.layers,
        ):
            torch.testing.assert_close(
                prefix_layer.keys, full_layer.keys[..., :3, :], rtol=1e-5, atol=1e-6,
            )
            torch.testing.assert_close(
                prefix_layer.values, full_layer.values[..., :3, :], rtol=1e-5, atol=1e-6,
            )
        torch.testing.assert_close(prefix.rope_deltas, full.rope_deltas, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
