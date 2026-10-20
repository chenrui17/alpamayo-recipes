# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for opt-in Stage-2 behavior and dependency patch verification.

The switch/patch checks use only the standard library. Forward/gradient checks
also require the recipe's normal torch/transformers/model dependencies.
"""

import os
import inspect
import sys
import unittest
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

from alpamayo1_5_sft.performance import stage2_optimizations as opt


def _frozen_model(**extra):
    return SimpleNamespace(
        cotrain_vlm=False,
        vlm=SimpleNamespace(parameters=lambda: [SimpleNamespace(requires_grad=False)]),
        **extra,
    )


class Stage2SwitchTests(unittest.TestCase):
    def test_default_does_not_import_optional_dependencies(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                opt.apply_stage2_optimizations(SimpleNamespace(cotrain_vlm=False)),
                {
                    "vlm_skip_head": False,
                    "liger_kernel": False,
                    "compile_expert": False,
                    "requested": {
                        "vlm_skip_head": False,
                        "liger_kernel": False,
                        "compile_expert": False,
                    },
                },
            )

    def test_joint_training_ignores_stage2_requests(self):
        env = {"APPLY_VLM_SKIP_HEAD": "1", "APPLY_LIGER_KERNEL": "1", "APPLY_COMPILE_EXPERT": "1"}
        with patch.dict(os.environ, env, clear=True):
            result = opt.apply_stage2_optimizations(SimpleNamespace(cotrain_vlm=True))
        self.assertEqual(result["skipped_reason"], "cotrain_vlm")
        self.assertFalse(result["vlm_skip_head"])
        self.assertFalse(result["liger_kernel"])
        self.assertFalse(result["compile_expert"])

    def test_skip_head_rejects_unsupported_forward(self):
        model = _frozen_model()
        model.vlm.forward = lambda input_ids: input_ids
        with patch.dict(os.environ, {"APPLY_VLM_SKIP_HEAD": "1"}, clear=True):
            with self.assertRaisesRegex(TypeError, "logits_to_keep"):
                opt.apply_stage2_optimizations(model)

    def test_requested_optimization_rejects_unfrozen_vlm(self):
        model = _frozen_model()
        model.vlm.parameters = lambda: [SimpleNamespace(requires_grad=True)]
        with patch.dict(os.environ, {"APPLY_LIGER_KERNEL": "1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "frozen"):
                opt.apply_stage2_optimizations(model)

    def test_compile_preserves_module_parameter_and_state_identity(self):
        calls = []
        executions = []

        class Module:
            def __init__(self):
                self.weight = object()

            def forward(self, value):
                return value + 1

            def state_dict(self):
                return {"weight": self.weight}

        def compile_forward(forward, **kwargs):
            self.assertTrue(inspect.ismethod(forward))
            calls.append((forward, kwargs))

            def compiled(*args, **kwargs):
                executions.append(forward.__self__)
                return forward(*args, **kwargs)

            return compiled

        torch_stub = ModuleType("torch")
        torch_stub.compile = compile_forward
        modules = {name: Module() for name in ("expert", "action_in_proj", "action_out_proj")}
        state = {name: module.state_dict() for name, module in modules.items()}
        model = _frozen_model(**modules)
        with patch.dict(os.environ, {"APPLY_COMPILE_EXPERT": "1"}, clear=True):
            with patch.dict(sys.modules, {"torch": torch_stub}):
                first = opt.apply_stage2_optimizations(model)
                second = opt.apply_stage2_optimizations(model)
        self.assertEqual(executions, [])
        for name, module in modules.items():
            self.assertIs(getattr(model, name), module)
            self.assertIs(module.weight, state[name]["weight"])
            self.assertEqual(module.state_dict(), state[name])
            self.assertFalse(hasattr(module, "_orig_mod"))
            self.assertEqual(module.forward(2), 3)
        self.assertEqual(len(executions), 3)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(kwargs == {"dynamic": True} for _, kwargs in calls))
        self.assertTrue(first["compile_expert"])
        self.assertEqual(first["compile"]["target"], "bound_forward")
        self.assertEqual(first["compile"]["execution"], "lazy_first_forward")
        self.assertEqual(first, second)

    def test_invalid_switch_is_not_silently_disabled(self):
        with patch.dict(os.environ, {"APPLY_COMPILE_EXPERT": "tru"}, clear=True):
            with self.assertRaisesRegex(ValueError, "APPLY_COMPILE_EXPERT"):
                opt.apply_stage2_optimizations(_frozen_model())

    def test_liger_noop_is_detected_and_loaded_swiglu_is_patched(self):
        def base_forward(self, value):
            return value

        def norm_forward(self, value):
            return value

        def swiglu_forward(self, value):
            return value

        def rope(*args):
            return args

        rope.__module__ = "liger_kernel.transformers.rope"

        def module(**extra):
            instance = SimpleNamespace(**extra)
            instance.forward = MethodType(base_forward, instance)
            return instance

        def stack():
            layer = SimpleNamespace(
                input_layernorm=module(),
                post_attention_layernorm=module(),
                self_attn=SimpleNamespace(q_norm=module(), k_norm=module()),
                mlp=module(gate_proj=object(), up_proj=object(), down_proj=object()),
            )
            return SimpleNamespace(norm=module(), layers=[layer], config=SimpleNamespace(hidden_act="silu"))

        def apply_liger(**kwargs):
            # Emulate Liger 0.7.0: RMSNorm/RoPE work, instantiated SwiGLU is omitted.
            self.assertFalse(kwargs["fused_linear_cross_entropy"])
            self.assertFalse(kwargs["cross_entropy"])
            norms, _ = opt._norms_and_mlps(kwargs["model"])
            for norm in norms:
                norm.forward = MethodType(norm_forward, norm)

        modules = {
            name: ModuleType(name)
            for name in (
                "liger_kernel",
                "liger_kernel.transformers",
                "liger_kernel.transformers.rms_norm",
                "liger_kernel.transformers.swiglu",
                "transformers",
                "transformers.models",
                "transformers.models.qwen3_vl",
                "transformers.models.qwen3_vl.modeling_qwen3_vl",
            )
        }
        modules["liger_kernel.transformers"].apply_liger_kernel_to_qwen3_vl = apply_liger
        modules["liger_kernel.transformers.rms_norm"].LigerRMSNorm = SimpleNamespace(forward=norm_forward)
        modules["liger_kernel.transformers.swiglu"].LigerSwiGLUMLP = SimpleNamespace(forward=swiglu_forward)
        qwen = modules["transformers.models.qwen3_vl.modeling_qwen3_vl"]
        qwen.apply_rotary_pos_emb = qwen.apply_rotary_pos_emb_vision = rope
        modules["transformers.models.qwen3_vl"].modeling_qwen3_vl = qwen

        with patch.dict(sys.modules, modules), patch.object(opt, "_text_stack", side_effect=lambda x: x):
            model = SimpleNamespace(vlm=stack(), expert=stack())
            result = opt._apply_liger(model, swiglu=True)
            self.assertEqual(result["vlm"], {"rms_norm_instances": 5, "swiglu_instances": 1})
            self.assertEqual(result["expert"], {"rms_norm_instances": 5, "swiglu_instances": 1})
            modules["liger_kernel.transformers"].apply_liger_kernel_to_qwen3_vl = lambda **kw: None
            with self.assertRaisesRegex(RuntimeError, "did not patch"):
                opt._apply_liger(SimpleNamespace(vlm=stack(), expert=stack()), swiglu=False)


class Stage2RealCompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest(f"torch unavailable: {exc}") from exc
        if not hasattr(torch, "compile"):
            raise unittest.SkipTest("torch.compile unavailable")
        cls.torch = torch

    def test_real_compile_forward_backward_and_checkpoint_roundtrip(self):
        import copy
        import io

        torch = self.torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cotrain_vlm = False
                self.vlm = torch.nn.Linear(2, 2).requires_grad_(False)
                self.expert = torch.nn.Linear(2, 2)
                self.action_in_proj = torch.nn.Linear(2, 2)
                self.action_out_proj = torch.nn.Linear(2, 2)

            def forward(self, value):
                return self.action_out_proj(self.expert(self.action_in_proj(value)))

        torch.manual_seed(42)
        model = Model()
        reference = copy.deepcopy(model)
        module_refs = {name: getattr(model, name) for name in ("expert", "action_in_proj", "action_out_proj")}
        parameter_refs = dict(model.named_parameters())
        original_keys = list(model.state_dict())
        real_compile = torch.compile
        # Real TorchDynamo compilation with a portable CPU backend. This test
        # verifies module/checkpoint behavior without requiring CUDA or Triton.
        with patch.dict(os.environ, {"APPLY_COMPILE_EXPERT": "1"}, clear=True):
            with patch.object(torch, "compile", side_effect=lambda fn, **kw: real_compile(fn, backend="eager", **kw)) as compiler:
                first = opt.apply_stage2_optimizations(model)
                second = opt.apply_stage2_optimizations(model)
        self.assertEqual(compiler.call_count, 3)
        self.assertEqual(first, second)
        for name, module in module_refs.items():
            self.assertIs(getattr(model, name), module)
        after_parameters = dict(model.named_parameters())
        self.assertEqual(list(after_parameters), list(parameter_refs))
        for name, parameter in parameter_refs.items():
            self.assertIs(after_parameters[name], parameter)
        self.assertEqual(list(model.state_dict()), original_keys)
        self.assertTrue(all("_orig_mod" not in name for name in original_keys))

        value = torch.randn(3, 2, requires_grad=True)
        reference_value = value.detach().clone().requires_grad_(True)
        expected = reference(reference_value)
        actual = model(value)
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(value.grad, reference_value.grad)
        for (name, parameter), (_, reference_parameter) in zip(model.named_parameters(), reference.named_parameters()):
            if reference_parameter.grad is None:
                self.assertIsNone(parameter.grad, name)
            else:
                torch.testing.assert_close(parameter.grad, reference_parameter.grad)
        self.assertEqual(list(model.state_dict()), original_keys)

        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        reloaded = Model()
        reloaded.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
        torch.testing.assert_close(reloaded(value.detach()), expected.detach())


class Stage2ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
        except ImportError as exc:
            raise unittest.SkipTest(f"Recipe dependencies unavailable: {exc}") from exc
        cls.torch = torch
        cls.forward = TrainableAlpamayoR1.forward

    def _model(self, cotrain=False):
        torch = self.torch
        model_forward = type(self).forward

        class VLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.25), requires_grad=cotrain)

            def forward(self, input_ids, labels, use_cache, logits_to_keep=None, **kwargs):
                self.call = {"labels": labels, "logits_to_keep": logits_to_keep, "use_cache": use_cache}
                cache = SimpleNamespace(layers=[], crop=lambda index: None)
                cache.condition = input_ids.float().sum() * self.weight
                return SimpleNamespace(loss=self.weight.square(), past_key_values=cache)

        class ActionIn(torch.nn.Module):
            def forward(self, noisy_x, timesteps):
                return noisy_x

        class Expert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, inputs_embeds, past_key_values, **kwargs):
                return SimpleNamespace(last_hidden_state=inputs_embeds * self.weight + past_key_values.condition)

        class Model(torch.nn.Module):
            forward = model_forward

            def __init__(self):
                super().__init__()
                self.cotrain_vlm = cotrain
                self.stop_grad_from_vlm = True
                self.vlm = VLM()
                self.expert = Expert()
                self.action_in_proj = ActionIn()
                self.action_out_proj = torch.nn.Identity()
                self.config = SimpleNamespace(traj_token_ids={"future_start": 7}, expert_non_causal_attention=False)
                self.action_space = SimpleNamespace(get_action_space_dims=lambda: (1,))
                self.diffusion = SimpleNamespace(compute_loss_from_pred=lambda training_data, pred: pred.square().mean())

            def fuse_traj_tokens(self, ids, traj):
                return ids

            def _process_traj_future_training(self, traj):
                return {"noisy_x": torch.ones(1, 1, 1), "timesteps": torch.zeros(1)}

            def _process_position_ids_qwen2_5_vl(self, *args):
                return torch.zeros(3, 1, 1, dtype=torch.long)

        return Model()

    def _run(self, enabled, cotrain=False):
        torch = self.torch
        model = self._model(cotrain)
        with patch.dict(os.environ, {"APPLY_VLM_SKIP_HEAD": str(int(enabled))}, clear=True):
            output = model(
                tokenized_data={"input_ids": torch.tensor([[7, 1, 2]])},
                labels_mask=torch.tensor([[True, False, True]]),
            )
            output.loss.backward()
        return model, output.loss.detach(), model.expert.weight.grad

    def test_frozen_skip_preserves_loss_and_expert_gradient(self):
        baseline, base_loss, base_grad = self._run(False)
        optimized, opt_loss, opt_grad = self._run(True)
        self.torch.testing.assert_close(base_loss, opt_loss, rtol=0, atol=0)
        self.torch.testing.assert_close(base_grad, opt_grad, rtol=0, atol=0)
        self.assertIsNotNone(baseline.vlm.call["labels"])
        self.assertIsNone(baseline.vlm.call["logits_to_keep"])
        self.assertIsNone(optimized.vlm.call["labels"])
        self.assertEqual(optimized.vlm.call["logits_to_keep"], 1)

    def test_joint_vlm_loss_and_gradient_are_unchanged(self):
        baseline, base_loss, base_grad = self._run(False, cotrain=True)
        enabled, enabled_loss, enabled_grad = self._run(True, cotrain=True)
        self.torch.testing.assert_close(base_loss, enabled_loss, rtol=0, atol=0)
        self.torch.testing.assert_close(base_grad, enabled_grad, rtol=0, atol=0)
        self.torch.testing.assert_close(baseline.vlm.weight.grad, enabled.vlm.weight.grad, rtol=0, atol=0)
        self.assertIsNotNone(enabled.vlm.call["labels"])
        self.assertIsNone(enabled.vlm.call["logits_to_keep"])


if __name__ == "__main__":
    unittest.main()
