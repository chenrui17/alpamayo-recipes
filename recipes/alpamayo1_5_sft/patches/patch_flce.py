#!/usr/bin/env python3
"""Idempotent patcher: add an APPLY_LIGER_FLCE fused-linear-cross-entropy path to
TrainableReasoningVLA in sft_base_model.py. Default behavior (env unset) unchanged.
"""
import sys

PATH = "/raid/charlie/alpamayo/alpamayo-recipes/recipes/alpamayo1_5_sft/models/sft_base_model.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

if "APPLY_LIGER_FLCE" in src:
    print("Already patched; nothing to do.")
    sys.exit(0)

orig = src

# --- 1. import os ---
imp_old = "import json\n"
imp_new = "import json\nimport os\n"
assert src.count(imp_old) >= 1, "import json anchor not found"
src = src.replace(imp_old, imp_new, 1)

# --- 2. insert _compute_next_token_loss_flce before def forward ---
method_anchor = (
    "            nan=0.0,\n"
    "        )\n"
    "        return loss\n"
    "\n"
    "    def forward(\n"
    "        self,\n"
    "        tokenized_data: dict[str, Any],\n"
)
assert src.count(method_anchor) == 1, "forward/return loss anchor not unique/found"

new_method = (
    "            nan=0.0,\n"
    "        )\n"
    "        return loss\n"
    "\n"
    "    @torch._dynamo.disable\n"
    "    def _compute_next_token_loss_flce(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "        labels: torch.Tensor,\n"
    "        labels_mask: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
    "        \"\"\"Fused-linear-cross-entropy variant of _compute_next_token_loss.\n"
    "\n"
    "        Fuses the lm_head projection and cross-entropy into a single Liger Triton\n"
    "        kernel, so the full (B, L, V) logits tensor is never materialized -- only\n"
    "        the selected (N, H) hidden states are kept. Mathematically equivalent (up\n"
    "        to floating-point error) to _compute_next_token_loss with token_mask=None.\n"
    "        \"\"\"\n"
    "        if labels_mask[:, 1:].sum() == 0:\n"
    "            return hidden_states.new_zeros(())\n"
    "        # Shift by one position (predict next token)\n"
    "        shift_hidden = hidden_states[..., :-1, :]\n"
    "        shift_labels = labels[..., 1:]\n"
    "        sel = labels_mask[:, 1:]\n"
    "        shift_hidden = shift_hidden[sel].contiguous()\n"
    "        shift_labels = shift_labels[sel].contiguous().to(shift_hidden.device)\n"
    "        flce_fn = getattr(self, \"_flce_loss_fn\", None)\n"
    "        if flce_fn is None:\n"
    "            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss\n"
    "\n"
    "            flce_fn = LigerFusedLinearCrossEntropyLoss(\n"
    "                ignore_index=IGNORE_INDEX, reduction=\"mean\"\n"
    "            )\n"
    "            self._flce_loss_fn = flce_fn\n"
    "        loss = flce_fn(\n"
    "            self.vlm.lm_head.weight,\n"
    "            shift_hidden,\n"
    "            shift_labels,\n"
    "            self.vlm.lm_head.bias,\n"
    "        )\n"
    "        return torch.nan_to_num(loss, nan=0.0)\n"
    "\n"
    "    def forward(\n"
    "        self,\n"
    "        tokenized_data: dict[str, Any],\n"
)
src = src.replace(method_anchor, new_method, 1)

# --- 3. branch the forward body ---
body_old = (
    "        # 3. vlm forward pass\n"
    "        outputs = self.vlm(input_ids=input_ids, labels=labels, **tokenized_data)\n"
    "\n"
    "        losses = {}\n"
)
assert src.count(body_old) == 1, "vlm forward-pass anchor not unique/found"
body_new = (
    "        # 3. vlm forward pass\n"
    "        #    APPLY_LIGER_FLCE=1 -> fused-linear-cross-entropy: fetch hidden states only\n"
    "        #    and fuse lm_head + CE per mask, never materializing (B, L, V) logits.\n"
    "        use_flce = os.environ.get(\"APPLY_LIGER_FLCE\", \"0\") == \"1\"\n"
    "        if use_flce:\n"
    "            base_outputs = self.vlm.model(input_ids=input_ids, **tokenized_data)\n"
    "            hidden_states = getattr(base_outputs, \"last_hidden_state\", None)\n"
    "            if hidden_states is None:\n"
    "                hidden_states = base_outputs[0]\n"
    "            outputs = None\n"
    "        else:\n"
    "            outputs = self.vlm(input_ids=input_ids, labels=labels, **tokenized_data)\n"
    "\n"
    "        losses = {}\n"
)
src = src.replace(body_old, body_new, 1)

fut_old = (
    "        losses[\"future_traj\"] = self._compute_next_token_loss(outputs, labels, traj_mask)\n"
)
assert src.count(fut_old) == 1, "future_traj anchor not unique/found"
fut_new = (
    "        if use_flce:\n"
    "            losses[\"future_traj\"] = self._compute_next_token_loss_flce(\n"
    "                hidden_states, labels, traj_mask\n"
    "            )\n"
    "        else:\n"
    "            losses[\"future_traj\"] = self._compute_next_token_loss(outputs, labels, traj_mask)\n"
)
src = src.replace(fut_old, fut_new, 1)

oth_old = (
    "        losses[\"others\"] = self._compute_next_token_loss(outputs, labels, labels != IGNORE_INDEX)\n"
)
assert src.count(oth_old) == 1, "others anchor not unique/found"
oth_new = (
    "        if use_flce:\n"
    "            losses[\"others\"] = self._compute_next_token_loss_flce(\n"
    "                hidden_states, labels, labels != IGNORE_INDEX\n"
    "            )\n"
    "        else:\n"
    "            losses[\"others\"] = self._compute_next_token_loss(outputs, labels, labels != IGNORE_INDEX)\n"
)
src = src.replace(oth_old, oth_new, 1)

ret_old = (
    "        # Replace the original loss\n"
    "        outputs.loss = sum(losses.values())\n"
    "\n"
    "        return ReasoningVLAOutput(\n"
    "            loss=outputs.loss,\n"
    "            logits=outputs.logits,\n"
    "        )\n"
)
assert src.count(ret_old) == 1, "return block anchor not unique/found"
ret_new = (
    "        # Replace the original loss\n"
    "        total_loss = sum(losses.values())\n"
    "\n"
    "        return ReasoningVLAOutput(\n"
    "            loss=total_loss,\n"
    "            logits=None if use_flce else outputs.logits,\n"
    "        )\n"
)
src = src.replace(ret_old, ret_new, 1)

assert src != orig, "no changes applied"

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

# sanity: byte-compile
import py_compile
py_compile.compile(PATH, doraise=True)
print("Patched and byte-compiled OK.")
