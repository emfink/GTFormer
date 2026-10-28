#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for the GraphTree Architecture.

Evaluates:
  - 7 Architectures (including Bidirectional Top-Down GraphTree & RoPE Transformer)
  - 8 Non-Math Long-Context Tasks (Code, Logs, Pointers, RAG, Scoping, XML, Decoys)
  - Up to 13 Sequence Lengths: N in [64 .. 4,194,304] (up to 8192x OOD extrapolation)
  - 4 Experimental Tiers (Main Matrix, Width Scaling Laws, Train-Length Horizon, Deep Seeds)

Features:
  - Automatic checkpoint/resume from `results_incremental.json`
  - Per-length console logging with latency and peak VRAM tracking
  - Automatic export of CSV summaries and publication-ready LaTeX booktabs tables
"""

import argparse
import csv
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# Global Benchmark Constants
# =====================================================================
TASKS: List[str] = [
    "override_qa",
    "unaligned_kv",
    "pointer_hop",
    "intrusion_fsm",
    "code_scope_shadow",
    "multi_needle_rag",
    "xml_bracket_check",
    "adversarial_decoys",
]

ARCHS: List[str] = [
    "GraphTree_Bidirectional",
    "GraphTree_BottomUp",
    "GraphTree_NoExpander",
    "Pure_Binary_Tree",
    "Unbounded_Tree_MLP",
    "Flat_Transformer_Sinusoidal",
    "Flat_Transformer_RoPE",
]

DEFAULT_EVAL_LENGTHS: List[int] = [
    64,
    256,
    512,
    1024,
    4096,
    16384,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
    4194304,
]

PRETTY_ARCH_NAMES: Dict[str, str] = {
    "GraphTree_Bidirectional": "GraphTree (Bidirectional: Up + Down)",
    "GraphTree_BottomUp": "GraphTree (Bottom-Up + Shortcuts)",
    "GraphTree_NoExpander": "GraphTree (Local Shift, No Expander)",
    "Pure_Binary_Tree": "Pure Binary Tree-GNN (No Shortcuts)",
    "Unbounded_Tree_MLP": "Unbounded Tree-GNN (No Convex Gate)",
    "Flat_Transformer_Sinusoidal": "Flat Transformer (Sinusoidal)",
    "Flat_Transformer_RoPE": "Flat Transformer (RoPE + AttnPool)",
}


# =====================================================================
# 1. Comprehensive 8-Task Long-Context Benchmark Suite (Vectorized)
# =====================================================================
class ComprehensiveLongContextSuite:
    """
    8 vectorized non-mathematical sequence reasoning tasks:
      1. 'override_qa'        : Chronological multi-register override (random register order)
      2. 'unaligned_kv'       : Multi-token [KEY_r, VAL_v] at odd binary-tree boundaries
      3. 'pointer_hop'        : 2-hop pointer dereferencing via [VAR_k, VAL_v] and [PTR_p, VAR_k]
      4. 'intrusion_fsm'      : Security log state tracking with mid-stream GLOBAL_RESET
      5. 'code_scope_shadow'  : Lexical block scoping (variables inside blocks revert on exit)
      6. 'multi_needle_rag'   : Multi-needle fact synthesis via [ENT_e, ATTR_v] and [QRY_p, ENT_e]
      7. 'xml_bracket_check'  : Hierarchical nested tag closure & corruption detection
      8. 'adversarial_decoys' : Unaligned KV recall under high-density syntactic decoy noise
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.num_filler = 200
        self.vocab_size = 320
        self.num_classes = 64  # Unified 64-class exact-match space (4^3)

    @torch.no_grad()
    def sample_batch(
        self, task_name: str, batch_size: int, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          docs:        (B, N) token IDs
          joint_label: (B,) exact-match class in [0 .. 63]
          slot_labels: (B, 11) [3 output slots (0..3), 5 var values (0..3 or -100), 3 ptr targets (0..4 or -100)]
        """
        dispatch = {
            "override_qa": self._sample_override_qa,
            "unaligned_kv": self._sample_unaligned_kv,
            "pointer_hop": self._sample_pointer_hop,
            "intrusion_fsm": self._sample_intrusion_fsm,
            "code_scope_shadow": self._sample_code_scope_shadow,
            "multi_needle_rag": self._sample_multi_needle_rag,
            "xml_bracket_check": self._sample_xml_bracket_check,
            "adversarial_decoys": self._sample_adversarial_decoys,
        }
        if task_name not in dispatch:
            raise ValueError(f"Unknown task: {task_name}")
        return dispatch[task_name](batch_size, seq_len)

    def _pack_labels(
        self,
        slots: torch.Tensor,
        var_vals: Optional[torch.Tensor] = None,
        ptr_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = slots.size(0)
        joint = slots[:, 0] * 16 + slots[:, 1] * 4 + slots[:, 2]
        if var_vals is None:
            var_vals = torch.full((bsz, 5), -100, dtype=torch.long, device=self.device)
        if ptr_targets is None:
            ptr_targets = torch.full((bsz, 3), -100, dtype=torch.long, device=self.device)
        packed_aux = torch.cat([slots, var_vals, ptr_targets], dim=1)  # (B, 11)
        return joint, packed_aux

    def _sample_odd_pos_in_segment(
        self, batch_size: int, seg_idx: int, seg_len: int
    ) -> torch.Tensor:
        """
        Samples an odd token index `pos = 2m + 1` strictly inside segment `seg_idx`
        such that both `pos` and `pos + 1` lie within `[seg_start, seg_end - 1]`.
        """
        seg_start = seg_idx * seg_len
        seg_end = (seg_idx + 1) * seg_len
        first_odd = seg_start + (1 - (seg_start % 2))
        max_odd = seg_end - 2
        num_odds = max(1, (max_odd - first_odd) // 2 + 1)
        return (
            first_odd
            + torch.randint(0, num_odds, (batch_size, 1), device=self.device) * 2
        )

    def _sample_override_qa(
        self, batch_size: int, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        docs = torch.randint(0, self.num_filler, (batch_size, seq_len), device=self.device)
        num_segs = 6
        seg_len = seq_len // num_segs
        final_vals = torch.zeros((batch_size, 3), dtype=torch.long, device=self.device)

        for wave in range(2):
            reg_order = torch.argsort(torch.rand(batch_size, 3, device=self.device), dim=-1)
            for s_idx in range(3):
                ev_idx = wave * 3 + s_idx
                reg = reg_order[:, s_idx : s_idx + 1]
                pos = ev_idx * seg_len + torch.randint(
                    0, seg_len, (batch_size, 1), device=self.device
                )
                val = torch.randint(0, 4, (batch_size, 1), device=self.device)
                tok = self.num_filler + reg * 4 + val
                docs.scatter_(1, pos, tok)
                final_vals.scatter_(1, reg, val)

        joint, slots = self._pack_labels(final_vals)
        return docs, joint, slots

    def _sample_unaligned_kv(
        self, batch_size: int, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Places 2-token [KEY_r, VAL_v] pairs in random key order at ODD indices (2m+1, 2m+2),
        splitting pairs across binary subtree boundaries while horizontal shift edges bind them.
        """
        docs = torch.randint(0, self.num_filler, (batch_size, seq_len), device=self.device)
        key_base = self.num_filler      # 200..202
        val_base = self.num_filler + 3  # 203..206

        num_distractors = max(2, seq_len // 16)
        dist_even_pos = (
            torch.randint(0, (seq_len - 4) // 2, (batch_size, num_distractors), device=self.device)
            * 2
        )
        dist_vals = val_base + torch.randint(
            0, 4, (batch_size, num_distractors), device=self.device
        )
        docs.scatter_(1, dist_even_pos, dist_vals)

        num_segs = 6
        seg_len = seq_len // num_segs
        final_vals = torch.zeros((batch_size, 3), dtype=torch.long, device=self.device)

        for wave in range(2):
            reg_order = torch.argsort(torch.rand(batch_size, 3, device=self.device), dim=-1)
            for s_idx in range(3):
                ev_idx = wave * 3 + s_idx
                reg = reg_order[:, s_idx : s_idx + 1]
                pos = self._sample_odd_pos_in_segment(batch_size, ev_idx, seg_len)

                val = torch.randint(0, 4, (batch_size, 1), device=self.device)
                docs.scatter_(1, pos, key_base + reg)
                docs.scatter_(1, pos + 1, val_base + val)
                final_vals.scatter_(1, reg, val)

        joint, slots = self._pack_labels(final_vals)
        return docs, joint, slots

    def _sample_pointer_hop(
        self, batch_size: int, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        2-Hop Pointer Dereferencing over 5 variables (VAR_0 .. VAR_4) and 3 pointers (PTR_0 .. PTR_2).
        """
        docs = torch.randint(0, self.num_filler, (batch_size, seq_len), device=self.device)
        var_key_base = self.num_filler  # 200..204 (5 variables)
        val_base = self.num_filler + 5  # 205..208 (4 values)
        ptr_base = self.num_filler + 9  # 209..211 (3 pointers)

        num_distractors = max(2, seq_len // 16)
        dist_even_pos = (
            torch.randint(0, (seq_len - 4) // 2, (batch_size, num_distractors), device=self.device)
            * 2
        )
        dist_vals = val_base + torch.randint(
            0, 4, (batch_size, num_distractors), device=self.device
        )
        docs.scatter_(1, dist_even_pos, dist_vals)

        num_segs = 8
        seg_len = seq_len // num_segs

        var_vals = torch.randint(0, 4, (batch_size, 5), device=self.device)
        var_order = torch.argsort(torch.rand(batch_size, 5, device=self.device), dim=-1)

        var_seg_indices = [0, 1, 3, 5, 6]
        for idx_v, seg_i in enumerate(var_seg_indices):
            pos = self._sample_odd_pos_in_segment(batch_size, seg_i, seg_len)
            v_id = var_order[:, idx_v : idx_v + 1]
            v_val = var_vals.gather(1, v_id)
            docs.scatter_(1, pos, var_key_base + v_id)
            docs.scatter_(1, pos + 1, val_base + v_val)

        ptr_seg_indices = [2, 4, 7]
        ptr_order = torch.argsort(torch.rand(batch_size, 3, device=self.device), dim=-1)
        ptr_targets = torch.argsort(torch.rand(batch_size, 5, device=self.device), dim=-1)[:, :3]

        for s_idx, seg_i in enumerate(ptr_seg_indices):
            pos = self._sample_odd_pos_in_segment(batch_size, seg_i, seg_len)
            p_id = ptr_order[:, s_idx : s_idx + 1]
            t_var = ptr_targets.gather(1, p_id)
            docs.scatter_(1, pos, ptr_base + p_id)
            docs.scatter_(1, pos + 1, var_key_base + t_var)

        final_slots = var_vals.gather(1, ptr_targets)
        joint, packed_aux = self._pack_labels(final_slots, var_vals, ptr_targets)
        return docs, joint, packed_aux

    def _sample_intrusion_fsm(
        self, batch_size: int, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        docs = torch.randint(0, self.num_filler, (batch_size, seq_len), device=self.device)
        num_segs = 7
        seg_len = seq_len // num_segs
        reset_tok = 250

        vals = torch.zeros((batch_size, 3), dtype=torch.long, device=self.device)
        reg_order_1 = torch.argsort(torch.rand(batch_size, 3, device=self.device), dim=-1)
        for s_idx in range(3):
            reg = reg_order_1[:, s_idx : s_idx + 1]
            pos = s_idx * seg_len + torch.randint(0, seg_len, (batch_size, 1), device=self.device)
            v = torch.randint(1, 4, (batch_size, 1), device=self.device)
            docs.scatter_(1, pos, self.num_filler + reg * 4 + v)
            vals.scatter_(1, reg, v)

        # Segment 3: 50% chance of GLOBAL_RESET wiping all registers to 0
        has_reset = torch.randint(0, 2, (batch_size, 1), device=self.device)
        pos_rst = 3 * seg_len + torch.randint(0, seg_len, (batch_size, 1), device=self.device)
        fallback = torch.randint(0, self.num_filler, (batch_size, 1), device=self.device)
        docs.scatter_(
            1, pos_rst, torch.where(has_reset == 1, torch.full_like(pos_rst, reset_tok), fallback)
        )
        vals = torch.where(has_reset == 1, torch.zeros_like(vals), vals)

        # Segments 4, 5: update 2 randomly chosen registers after the potential reset
        reg_order_2 = torch.argsort(torch.rand(batch_size, 3, device=self.device), dim=-1)[:, :2]
        for s_idx in range(2):
            reg = reg_order_2[:, s_idx : s_idx + 1]
            pos = (4 + s_idx) * seg_len + torch.randint(
                0, seg_len, (batch_size, 1), device=self.device
            )
            v = torch.randint(1, 4, (batch_size, 1), device=self.device)
            docs.scatter_(1, pos, self.num_filler + reg * 4 + v)
            vals.scatter_(1, reg, v)

        joint, slots = self._pack_labels(vals)
        return docs, joint, slots

    def _sample_code_scope_shadow(
        self, batch_size: int, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulates lexical variable shadowing in source code with SCOPE_ENTER,
        SCOPE_EXIT (revert), and SCOPE_COMMIT (keep).
        """
        docs = torch.randint(0, self.num_filler, (batch_size, seq_len), device=self.device)
        num_segs = 8
        seg_len = seq_len // num_segs
        scope_enter_tok = 260
        scope_exit_revert_tok = 261
        scope_commit_tok = 262

        global_vals = torch.zeros((batch_size, 3), dtype=torch.long, device=self.device)
        order_g = torch.argsort(torch.rand(batch_size, 3, device=self.device), dim=-1)
        for s_idx in range(3):
            reg = order_g[:, s_idx : s_idx + 1]
            pos = s_idx * seg_len + torch.randint(0, seg_len, (batch_size, 1), device=self.device)
            v = torch.randint(0, 4, (batch_size, 1), device=self.device)
            docs.scatter_(1, pos, self.num_filler + reg * 4 + v)
            global_vals.scatter_(1, reg, v)

        pos_enter = 3 * seg_len + torch.randint(0, seg_len, (batch_size, 1), device=self.device)
        docs.scatter_(1, pos_enter, torch.full_like(pos_enter, scope_enter_tok))

        inner_vals = global_vals.clone()
        order_in = torch.argsort(torch.rand(batch_size, 3, device=self.device), dim=-1)
        for s_idx in range(3):
            reg = order_in[:, s_idx : s_idx + 1]
            pos = (4 + s_idx) * seg_len + torch.randint(
                0, seg_len, (batch_size, 1), device=self.device
            )
            v = torch.randint(0, 4, (batch_size, 1), device=self.device)
            docs.scatter_(1, pos, self.num_filler + reg * 4 + v)
            inner_vals.scatter_(1, reg, v)

        is_revert = torch.randint(0, 2, (batch_size, 1), device=self.device)
        pos_exit = 7 * seg_len + torch.randint(0, seg_len, (batch_size, 1), device=self.device)
        end_tok = torch.where(
            is_revert == 1,
            torch.full_like(pos_exit, scope_exit_revert_tok),
            torch.full_like(pos_exit, scope_commit_tok),
        )
        docs.scatter_(1, pos_exit, end_tok)

        final_vals = torch.where(is_revert == 1, global_vals, inner_vals)
        joint, slots = self._pack_labels(final_vals)
        return docs, joint, slots

    def _sample_multi_needle_rag(
        self, batch_size: int, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-Needle RAG Synthesis under Distractor Noise.
        """
        docs = torch.randint(0, self.num_filler, (batch_size, seq_len), device=self.device)
        ent_base = self.num_filler       # 200..204 (5 entities)
        attr_base = self.num_filler + 5  # 205..208 (4 attribute values)
        qry_base = self.num_filler + 9   # 209..211 (3 query slot markers)

        num_distractors = max(2, seq_len // 16)
        dist_even_pos = (
            torch.randint(0, (seq_len - 4) // 2, (batch_size, num_distractors), device=self.device)
            * 2
        )
        dist_attrs = attr_base + torch.randint(
            0, 4, (batch_size, num_distractors), device=self.device
        )
        docs.scatter_(1, dist_even_pos, dist_attrs)

        num_segs = 8
        seg_len = seq_len // num_segs

        ent_vals = torch.randint(0, 4, (batch_size, 5), device=self.device)
        ent_order = torch.argsort(torch.rand(batch_size, 5, device=self.device), dim=-1)

        fact_segs = [0, 1, 2, 4, 6]
        for idx_f, seg_i in enumerate(fact_segs):
            pos = self._sample_odd_pos_in_segment(batch_size, seg_i, seg_len)
            e_id = ent_order[:, idx_f : idx_f + 1]
            v_id = ent_vals.gather(1, e_id)
            docs.scatter_(1, pos, ent_base + e_id)
            docs.scatter_(1, pos + 1, attr_base + v_id)

        qry_segs = [3, 5, 7]
        qry_order = torch.argsort(torch.rand(batch_size, 3, device=self.device), dim=-1)
        qry_targets = torch.argsort(torch.rand(batch_size, 5, device=self.device), dim=-1)[:, :3]

        for s_idx, seg_i in enumerate(qry_segs):
            pos = self._sample_odd_pos_in_segment(batch_size, seg_i, seg_len)
            q_slot = qry_order[:, s_idx : s_idx + 1]
            t_ent = qry_targets.gather(1, q_slot)
            docs.scatter_(1, pos, qry_base + q_slot)
            docs.scatter_(1, pos + 1, ent_base + t_ent)

        final_slots = ent_vals.gather(1, qry_targets)
        joint, packed_aux = self._pack_labels(final_slots, ent_vals, qry_targets)
        return docs, joint, packed_aux

    def _sample_xml_bracket_check(
        self, batch_size: int, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hierarchical XML/Code Tag Tracking across 3 independent subsystems (0, 1, 2).
        """
        docs = torch.randint(0, self.num_filler, (batch_size, seq_len), device=self.device)
        open_base = self.num_filler        # 200..211 (3 subsystems x 4 tag types)
        close_base = self.num_filler + 12  # 212..223 (matching close tags)

        num_segs = 6
        seg_len = seq_len // num_segs
        final_status = torch.zeros((batch_size, 3), dtype=torch.long, device=self.device)

        sub_order = torch.argsort(torch.rand(batch_size, 3, device=self.device), dim=-1)
        for s_idx in range(3):
            sub = sub_order[:, s_idx : s_idx + 1]
            tag_type = torch.randint(1, 4, (batch_size, 1), device=self.device)
            is_closed = torch.randint(0, 2, (batch_size, 1), device=self.device)

            pos_open = s_idx * seg_len + torch.randint(
                0, seg_len, (batch_size, 1), device=self.device
            )
            pos_close = (s_idx + 3) * seg_len + torch.randint(
                0, seg_len, (batch_size, 1), device=self.device
            )

            docs.scatter_(1, pos_open, open_base + sub * 4 + tag_type)
            close_tok = torch.where(
                is_closed == 1,
                close_base + sub * 4 + tag_type,
                torch.randint(0, self.num_filler, (batch_size, 1), device=self.device),
            )
            docs.scatter_(1, pos_close, close_tok)

            status = torch.where(is_closed == 1, torch.zeros_like(tag_type), tag_type)
            final_status.scatter_(1, sub, status)

        joint, slots = self._pack_labels(final_status)
        return docs, joint, slots

    def _sample_adversarial_decoys(
        self, batch_size: int, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        High-Density Adversarial Distractor Stress Test.
        """
        docs, joint, slots = self._sample_unaligned_kv(batch_size, seq_len)
        val_base = self.num_filler + 3         # 203..206
        decoy_key_base = self.num_filler + 40  # 240..242

        num_decoys = max(4, seq_len // 16)
        decoy_pos = (
            torch.randint(0, (seq_len - 4) // 2, (batch_size, num_decoys), device=self.device) * 2
        )
        curr_toks = docs.gather(1, decoy_pos)
        curr_next = docs.gather(1, decoy_pos + 1)
        can_place = (curr_toks < self.num_filler) & (curr_next < self.num_filler)

        fake_k = decoy_key_base + torch.randint(0, 3, (batch_size, num_decoys), device=self.device)
        fake_v = val_base + torch.randint(0, 4, (batch_size, num_decoys), device=self.device)

        docs.scatter_(1, decoy_pos, torch.where(can_place, fake_k, curr_toks))
        docs.scatter_(1, decoy_pos + 1, torch.where(can_place, fake_v, curr_next))
        return docs, joint, slots


# =====================================================================
# 2. Model Zoo (7 Architectures for Complete Paper Ablation)
# =====================================================================
class MultiHeadOutputHead(nn.Module):
    """
    Unified readout head with differentiable 2-hop relational dereferencing (`bmm(P, V)`)
    and factored 64-class exact-match composition.
    """

    def __init__(self, d_model: int, num_classes: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )
        self.slot_heads = nn.Linear(d_model, 12)  # Direct 3 slots x 4 values
        self.var_head = nn.Linear(d_model, 20)    # 5 latent variables/entities x 4 values
        self.ptr_head = nn.Linear(d_model, 15)    # 3 pointers/queries x 5 target variables
        self.rel_gate = nn.Linear(d_model, 3)     # Blends direct 1-hop vs relational 2-hop slots
        self.joint_head = nn.Linear(d_model, num_classes)

    def forward(
        self, root_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        feat = self.mlp(root_vec)

        direct_slots = self.slot_heads(feat).view(-1, 3, 4)
        var_logits = self.var_head(feat).view(-1, 5, 4)
        ptr_logits = self.ptr_head(feat).view(-1, 3, 5)

        # Differentiable 2-hop relational join: P(Slot_p = v) = sum_k P(Ptr_p = k) * P(Var_k = v)
        ptr_probs = F.softmax(ptr_logits, dim=-1)
        var_probs = F.softmax(var_logits, dim=-1)
        hop_probs = torch.bmm(ptr_probs, var_probs)  # (B, 3, 4)
        hop_logits = torch.log(hop_probs.clamp(min=1e-7))

        g_rel = torch.sigmoid(self.rel_gate(feat)).unsqueeze(-1)  # (B, 3, 1)
        slot_logits = (1.0 - g_rel) * direct_slots + g_rel * hop_logits

        s0 = slot_logits[:, 0, :, None, None]
        s1 = slot_logits[:, 1, None, :, None]
        s2 = slot_logits[:, 2, None, None, :]
        factored_joint = (s0 + s1 + s2).reshape(-1, 64)

        joint_logits = self.joint_head(feat) + factored_joint
        return joint_logits, (slot_logits, var_logits, ptr_logits)


class GraphTreeCell(nn.Module):
    """
    Scale-invariant hierarchical graph message-passing cell.
    """

    def __init__(
        self,
        d_model: int,
        use_expander: bool = True,
        use_convex_gate: bool = True,
        use_top_down: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_expander = use_expander
        self.use_convex_gate = use_convex_gate
        self.use_top_down = use_top_down

        if use_expander:
            self.exp_gate = nn.Linear(d_model * 2, d_model)
            nn.init.constant_(self.exp_gate.bias, -4.0)
            self.exp_proj = nn.Linear(d_model, d_model, bias=False)
            nn.init.eye_(self.exp_proj.weight)

        if use_convex_gate:
            self.score_net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.cross_mod = nn.Linear(d_model * 2, d_model)
            self.override_bias = nn.Parameter(torch.full((1, 1, d_model), 0.15))
        else:
            self.unbounded_mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
            )

        if use_top_down:
            self.td_parent_proj = nn.Linear(d_model, d_model, bias=False)
            self.td_node_proj = nn.Linear(d_model, d_model, bias=False)
            self.td_gate = nn.Sequential(
                nn.LayerNorm(d_model * 3),
                nn.Linear(d_model * 3, d_model),
                nn.Sigmoid(),
            )
            self.td_update = nn.Sequential(
                nn.LayerNorm(d_model * 3),
                nn.Linear(d_model * 3, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.td_score_net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.td_cross_mod = nn.Linear(d_model * 2, d_model)

    def bottom_up_step(self, h_curr: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = h_curr.shape

        if self.use_expander and seq_len > 2:
            h_peer = torch.roll(h_curr, shifts=seq_len // 2, dims=1)
            if self.use_convex_gate:
                s_c = self.score_net(h_curr)
                s_p = self.score_net(h_peer)
                g_e = torch.sigmoid(
                    (s_p - s_c) + self.exp_gate(torch.cat([h_curr, h_peer], dim=-1))
                )
            else:
                g_e = torch.sigmoid(self.exp_gate(torch.cat([h_curr, h_peer], dim=-1)))
            h_curr = (1.0 - g_e) * h_curr + g_e * self.exp_proj(h_peer)

        h_l = h_curr[:, 0::2, :]
        h_r = h_curr[:, 1::2, :]

        if self.use_convex_gate:
            s_l = self.score_net(h_l)
            s_r = self.score_net(h_r)
            c_m = torch.tanh(self.cross_mod(torch.cat([h_l, h_r], dim=-1)))
            g = torch.sigmoid((s_r - s_l) + c_m + self.override_bias)
            return g * h_r + (1.0 - g) * h_l
        return self.unbounded_mlp(torch.cat([h_l, h_r], dim=-1))

    def top_down_refine_nodes(
        self, h_nodes: torch.Tensor, h_root: torch.Tensor
    ) -> torch.Tensor:
        """
        Broadcasts the global root context `h_root` (B, 1, D) down to coarse subtree nodes
        and performs multiplicative bilinear conditioning + convex-gated tree reduction.
        """
        p_proj = self.td_parent_proj(h_root).expand_as(h_nodes)
        n_proj = self.td_node_proj(h_nodes)
        bilinear = n_proj * p_proj
        cat_feat = torch.cat([h_nodes, p_proj, bilinear], dim=-1)
        g_td = self.td_gate(cat_feat)
        curr = h_nodes + g_td * self.td_update(cat_feat)

        while curr.size(1) > 1:
            h_l = curr[:, 0::2, :]
            h_r = curr[:, 1::2, :]
            s_l = self.td_score_net(h_l)
            s_r = self.td_score_net(h_r)
            c_m = torch.tanh(self.td_cross_mod(torch.cat([h_l, h_r], dim=-1)))
            g = torch.sigmoid((s_r - s_l) + c_m)
            curr = g * h_r + (1.0 - g) * h_l

        return curr.squeeze(1)


class GraphTreeBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 64,
        d_model: int = 128,
        use_local_shift: bool = True,
        use_expander: bool = True,
        use_convex_gate: bool = True,
        use_top_down: bool = False,
    ):
        super().__init__()
        self.use_local_shift = use_local_shift
        self.use_top_down = use_top_down
        self.embedding = nn.Embedding(vocab_size, d_model)

        if use_local_shift:
            self.bind_l = nn.Linear(d_model, d_model, bias=False)
            self.bind_r = nn.Linear(d_model, d_model, bias=False)
            self.local_bind = nn.Sequential(
                nn.LayerNorm(d_model * 3),
                nn.Linear(d_model * 3, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
        else:
            self.local_bind = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )

        self.cell = GraphTreeCell(
            d_model=d_model,
            use_expander=use_expander,
            use_convex_gate=use_convex_gate,
            use_top_down=use_top_down,
        )

        if use_top_down:
            self.root_merge = nn.Linear(d_model * 2, d_model)
            nn.init.normal_(self.root_merge.weight, std=0.02)
            nn.init.zeros_(self.root_merge.bias)

        self.readout = MultiHeadOutputHead(d_model, num_classes)

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h = self.embedding(tokens)
        if self.use_local_shift:
            h_next = F.pad(h[:, 1:, :], (0, 0, 0, 1))
            h_inter = F.gelu(self.bind_l(h)) * self.bind_r(h_next)
            h_leaves = self.local_bind(torch.cat([h, h_next, h_inter], dim=-1))
        else:
            h_leaves = self.local_bind(h)

        # Phase 1: Bottom-Up Tree Reduction in O(log2 N) steps
        curr = h_leaves
        h_mid = curr if curr.size(1) <= 16 else None
        while curr.size(1) > 1:
            curr = self.cell.bottom_up_step(curr)
            if curr.size(1) == 16:
                h_mid = curr
        h_root = curr  # (B, 1, D)

        # Phase 2: Scale-Invariant Top-Down Broadcast & Subtree Resolution
        if self.use_top_down:
            if h_mid is None:
                h_mid = h_leaves
            h_td_summary = self.cell.top_down_refine_nodes(h_mid, h_root)
            root_flat = h_root.squeeze(1)
            final_vec = root_flat + self.root_merge(torch.cat([root_flat, h_td_summary], dim=-1))
        else:
            final_vec = h_root.squeeze(1)

        return self.readout(final_vec)


class FlatTransformerBaseline(nn.Module):
    """
    Supports both Sinusoidal Positional Encodings (`use_rope=False`) and
    Rotary Position Embeddings + Gated Attention Pooling (`use_rope=True`).
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        use_rope: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.use_rope = use_rope
        self.embedding = nn.Embedding(vocab_size, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.attn_pool_score = nn.Linear(d_model, d_model)
        self.readout = MultiHeadOutputHead(d_model, num_classes)

    def _pos_enc(self, seq_len: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=device)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros((1, seq_len, self.d_model), device=device)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe

    def _apply_rope_embedding(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        pos = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float, device=x.device)
            * (-math.log(10000.0) / dim)
        )
        sin = torch.sin(pos * div).unsqueeze(0)
        cos = torch.cos(pos * div).unsqueeze(0)
        x1 = x[:, :, 0::2]
        x2 = x[:, :, 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).reshape(
            bsz, seq_len, dim
        )

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        _, seq_len = tokens.shape
        x = self.embedding(tokens)
        if self.use_rope:
            x = self._apply_rope_embedding(x)
            x = self.encoder(x)
            w = torch.softmax(self.attn_pool_score(x), dim=1)
            pooled = (w * x).sum(dim=1)
        else:
            x = x + self._pos_enc(seq_len, tokens.device)
            x = self.encoder(x)
            pooled = x.mean(dim=1)
        return self.readout(pooled)


def build_model(
    arch_name: str, suite: ComprehensiveLongContextSuite, d_model: int = 128
) -> nn.Module:
    v_sz, n_cls = suite.vocab_size, suite.num_classes
    configs = {
        "GraphTree_Bidirectional": dict(
            use_local_shift=True, use_expander=True, use_convex_gate=True, use_top_down=True
        ),
        "GraphTree_BottomUp": dict(
            use_local_shift=True, use_expander=True, use_convex_gate=True, use_top_down=False
        ),
        "GraphTree_NoExpander": dict(
            use_local_shift=True, use_expander=False, use_convex_gate=True, use_top_down=False
        ),
        "Pure_Binary_Tree": dict(
            use_local_shift=False, use_expander=False, use_convex_gate=True, use_top_down=False
        ),
        "Unbounded_Tree_MLP": dict(
            use_local_shift=True, use_expander=False, use_convex_gate=False, use_top_down=False
        ),
    }
    if arch_name in configs:
        return GraphTreeBackbone(v_sz, n_cls, d_model, **configs[arch_name])
    if arch_name == "Flat_Transformer_Sinusoidal":
        return FlatTransformerBaseline(v_sz, n_cls, d_model, use_rope=False)
    if arch_name == "Flat_Transformer_RoPE":
        return FlatTransformerBaseline(v_sz, n_cls, d_model, use_rope=True)
    raise ValueError(f"Unknown architecture: {arch_name}")


# =====================================================================
# 3. Training & Multi-Length Evaluation Engine
# =====================================================================
def get_train_lengths(max_train_len: int) -> List[int]:
    all_candidates = [32, 64, 128, 256, 512, 1024]
    return [length for length in all_candidates if length <= max_train_len]


def train_single_run(
    model: nn.Module,
    suite: ComprehensiveLongContextSuite,
    task_name: str,
    steps: int,
    lr: float,
    max_train_len: int,
    run_tag: str,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.05
    )
    model.train()

    train_lengths = get_train_lengths(max_train_len)
    short_lengths = train_lengths[: min(2, len(train_lengths))]
    warmup_steps = steps // 4
    t0 = time.time()

    for step in range(1, steps + 1):
        if step <= warmup_steps:
            seq_len = short_lengths[step % len(short_lengths)]
        else:
            seq_len = train_lengths[step % len(train_lengths)]

        bs = 64 if seq_len <= 128 else (32 if seq_len <= 512 else 16)
        docs, joint_targets, aux_targets = suite.sample_batch(task_name, bs, seq_len)

        joint_logits, (slot_logits, var_logits, ptr_logits) = model(docs)
        loss_joint = F.cross_entropy(joint_logits, joint_targets)

        slot_targets = aux_targets[:, 0:3]
        loss_slots = (
            F.cross_entropy(slot_logits[:, 0, :], slot_targets[:, 0])
            + F.cross_entropy(slot_logits[:, 1, :], slot_targets[:, 1])
            + F.cross_entropy(slot_logits[:, 2, :], slot_targets[:, 2])
        ) / 3.0
        loss = loss_joint + 0.5 * loss_slots

        # On 2-hop relational tasks, supervise latent variable & pointer tables during training
        if aux_targets[0, 3].item() >= 0:
            var_targets = aux_targets[:, 3:8]
            ptr_targets = aux_targets[:, 8:11]
            loss_vars = F.cross_entropy(var_logits.reshape(-1, 4), var_targets.reshape(-1))
            loss_ptrs = F.cross_entropy(ptr_logits.reshape(-1, 5), ptr_targets.reshape(-1))
            loss = loss + 1.0 * (loss_vars + loss_ptrs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % max(1, steps // 3) == 0 or step == steps:
            acc = (joint_logits.argmax(dim=-1) == joint_targets).float().mean().item() * 100.0
            print(
                f"    [{run_tag}] Step {step:4d}/{steps} | Loss: {loss.item():.4f} "
                f"| TrainAcc(N={seq_len}): {acc:6.2f}% | {time.time() - t0:.1f}s"
            )
    return time.time() - t0


@torch.inference_mode()
def evaluate_single_run(
    model: nn.Module,
    suite: ComprehensiveLongContextSuite,
    task_name: str,
    eval_lengths: List[int],
    is_flat_transformer: bool,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    out: Dict[str, Dict[str, float]] = {}
    use_cuda = suite.device.type == "cuda"

    for seq_len in eval_lengths:
        if is_flat_transformer and seq_len > 4096:
            out[str(seq_len)] = {"acc": -1.0, "ms": -1.0, "vram_mb": -1.0}
            continue

        if seq_len <= 1024:
            bs, batches = 32, 6
        elif seq_len <= 4096:
            bs, batches = 16, 8
        elif seq_len <= 16384:
            bs, batches = 8, 8
        elif seq_len <= 65536:
            bs, batches = 4, 8
        else:
            bs, batches = 1, 16  # Batch size 1 for extreme lengths (N >= 131,072)

        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        correct, total = 0, 0
        t0 = time.time()
        try:
            for _ in range(batches):
                docs, joint_targets, _ = suite.sample_batch(task_name, bs, seq_len)

                with torch.autocast(
                    device_type=suite.device.type,
                    dtype=torch.float16,
                    enabled=use_cuda,
                ):
                    joint_logits, _ = model(docs)

                del docs
                preds = joint_logits.argmax(dim=-1)
                del joint_logits

                correct += (preds == joint_targets).sum().item()
                total += bs

                del preds, joint_targets
                if use_cuda:
                    torch.cuda.empty_cache()

            if use_cuda:
                torch.cuda.synchronize()
            ms_per_seq = (time.time() - t0) * 1000.0 / total
            vram_mb = (
                torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if use_cuda else 0.0
            )
            out[str(seq_len)] = {
                "acc": round(100.0 * correct / total, 2),
                "ms": round(ms_per_seq, 2),
                "vram_mb": round(vram_mb, 1),
            }
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if use_cuda:
                    torch.cuda.empty_cache()
                out[str(seq_len)] = {"acc": -1.0, "ms": -1.0, "vram_mb": -1.0}
            else:
                raise e

    return out


# =====================================================================
# 4. Incremental Checkpointing, CSV & LaTeX Booktabs Exporter
# =====================================================================
def format_eval_line(eval_dict: Dict[str, Dict[str, float]], eval_lengths: List[int]) -> str:
    parts = []
    for length in eval_lengths:
        m = eval_dict.get(str(length), {"acc": -1.0})
        if length >= 1048576:
            tag = f"{length // 1048576}M"
        elif length >= 1024:
            tag = f"{length // 1024}k"
        else:
            tag = str(length)

        if m["acc"] < 0:
            parts.append(f"N={tag}:   OOM")
        else:
            parts.append(f"N={tag}:{m['acc']:6.1f}%")

    m65 = eval_dict.get("65536", {"ms": -1.0, "vram_mb": -1.0})
    m131 = eval_dict.get("131072", {"ms": -1.0, "vram_mb": -1.0})
    tail = f" | 65k:{m65['ms']:.1f}ms | 131k:{m131['ms']:.1f}ms ({m131['vram_mb']:.0f}MB)"
    return " | ".join(parts) + tail


def _compute_stats(m_list: List[Dict[str, float]]) -> Tuple[float, float, float, float]:
    accs = [m["acc"] for m in m_list if m["acc"] >= 0]
    mss = [m["ms"] for m in m_list if m["ms"] >= 0]
    vrams = [m["vram_mb"] for m in m_list if m["vram_mb"] >= 0]
    if not accs:
        return -1.0, 0.0, -1.0, -1.0
    mean_acc = sum(accs) / len(accs)
    std_acc = math.sqrt(sum((x - mean_acc) ** 2 for x in accs) / len(accs))
    mean_ms = sum(mss) / len(mss)
    mean_vram = sum(vrams) / len(vrams)
    return round(mean_acc, 2), round(std_acc, 2), round(mean_ms, 2), round(mean_vram, 1)


def export_all_reports(
    raw_records: List[Dict[str, Any]],
    tasks: List[str],
    archs: List[str],
    eval_lengths: List[int],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "results_incremental.json")
    csv_main_path = os.path.join(out_dir, "results_summary.csv")
    csv_scaling_path = os.path.join(out_dir, "scaling_laws_summary.csv")
    tex_path = os.path.join(out_dir, "paper_tables.tex")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, indent=2)

    grouped: Dict[Tuple[str, str, str, int, int, int], List[Dict[str, float]]] = {}
    for rec in raw_records:
        tier = rec.get("tier", "tier1_main")
        t = rec["task"]
        a = rec["arch"]
        d_m = rec.get("d_model", 128)
        m_tr = rec.get("max_train_len", 512)
        for length_str, metrics in rec["eval"].items():
            key = (tier, t, a, d_m, m_tr, int(length_str))
            grouped.setdefault(key, []).append(metrics)

    scaling_rows = []
    for (tier, t, a, d_m, m_tr, length), m_list in grouped.items():
        if tier in ("tier1_main", "tier4_deep_seeds") and d_m == 128 and m_tr == 512:
            continue
        mean_acc, std_acc, mean_ms, mean_vram = _compute_stats(m_list)
        scaling_rows.append(
            {
                "tier": tier,
                "task": t,
                "arch": a,
                "d_model": d_m,
                "max_train_len": m_tr,
                "seq_len": length,
                "n_seeds": len(m_list),
                "acc_mean": mean_acc,
                "acc_std": std_acc,
                "latency_ms": mean_ms,
                "peak_vram_mb": mean_vram,
            }
        )

    # Merge tier1_main + tier4_deep_seeds for the unified main table
    merged_main_groups: Dict[Tuple[str, str, int], List[Dict[str, float]]] = {}
    for rec in raw_records:
        if rec.get("d_model", 128) == 128 and rec.get("max_train_len", 512) == 512:
            t, a = rec["task"], rec["arch"]
            for length_str, metrics in rec["eval"].items():
                merged_main_groups.setdefault((t, a, int(length_str)), []).append(metrics)

    unified_main_rows = []
    for (t, a, length), m_list in merged_main_groups.items():
        mean_acc, std_acc, mean_ms, mean_vram = _compute_stats(m_list)
        unified_main_rows.append(
            {
                "task": t,
                "arch": a,
                "seq_len": length,
                "n_seeds": len(m_list),
                "acc_mean": mean_acc,
                "acc_std": std_acc,
                "latency_ms": mean_ms,
                "peak_vram_mb": mean_vram,
            }
        )

    fieldnames_main = [
        "task", "arch", "seq_len", "n_seeds", "acc_mean", "acc_std", "latency_ms", "peak_vram_mb"
    ]
    with open(csv_main_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_main)
        writer.writeheader()
        writer.writerows(unified_main_rows)

    fieldnames_scaling = [
        "tier", "task", "arch", "d_model", "max_train_len", "seq_len",
        "n_seeds", "acc_mean", "acc_std", "latency_ms", "peak_vram_mb",
    ]
    with open(csv_scaling_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_scaling)
        writer.writeheader()
        writer.writerows(scaling_rows)

    # Build LaTeX Booktabs Tables
    lookup = {(r["task"], r["arch"], r["seq_len"]): r for r in unified_main_rows}
    tex_lines = [
        "% Auto-generated LaTeX Booktabs Tables for GraphTree Paper",
        "% Requires \\usepackage{booktabs}",
        "",
    ]

    for t in tasks:
        tex_lines.append("\\begin{table*}[htbp]")
        tex_lines.append("\\centering")
        tex_lines.append("\\scriptsize")
        col_spec = "l" + ("c" * len(eval_lengths))
        tex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        tex_lines.append("\\toprule")
        len_headers = " & ".join(
            f"$N={length}$" + (" (ID)" if length <= 512 else " (OOD)")
            for length in eval_lengths
        )
        tex_lines.append(f"Architecture & {len_headers} \\\\")
        tex_lines.append("\\midrule")

        for a in archs:
            cells = []
            for length in eval_lengths:
                row = lookup.get((t, a, length))
                if not row:
                    cells.append("--")
                elif row["acc_mean"] < 0:
                    cells.append("OOM")
                else:
                    cells.append(f"${row['acc_mean']:.1f} \\pm {row['acc_std']:.1f}\\%$")
            tex_lines.append(f"{PRETTY_ARCH_NAMES.get(a, a)} & " + " & ".join(cells) + " \\\\")

        tex_lines.append("\\bottomrule")
        tex_lines.append("\\end{tabular}")
        max_eval = max(eval_lengths)
        tex_lines.append(
            f"\\caption{{Zero-shot length extrapolation accuracy (mean $\\pm$ std) on task "
            f"\\texttt{{{t}}}. Trained exclusively on $N \\le 512$; evaluated up to "
            f"$N = {max_eval:,}$ (${max_eval // 512}\\times$ extrapolation).}}"
        )
        tex_lines.append(f"\\label{{tab:{t}}}")
        tex_lines.append("\\end{table*}\n")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))


# =====================================================================
# 5. Multi-Tier Experiment Orchestrator
# =====================================================================
def build_experiment_queue(base_seeds: int, steps: int) -> List[Dict[str, Any]]:
    queue: List[Dict[str, Any]] = []

    # TIER 1: Main 8 Tasks x 7 Architectures x base_seeds
    for seed_idx in range(base_seeds):
        seed = 100 + seed_idx
        for t in TASKS:
            for a in ARCHS:
                queue.append(
                    {
                        "tier": "tier1_main",
                        "task": t,
                        "arch": a,
                        "d_model": 128,
                        "max_train_len": 512,
                        "seed": seed,
                        "steps": steps,
                    }
                )

    # TIER 2: Width & Capacity Scaling Laws (d_model in [64, 96, 192, 256])
    scaling_tasks = ["unaligned_kv", "pointer_hop", "code_scope_shadow", "multi_needle_rag"]
    scaling_archs = [
        "GraphTree_Bidirectional",
        "GraphTree_BottomUp",
        "Pure_Binary_Tree",
        "Flat_Transformer_RoPE",
    ]
    for d_m in [64, 96, 192, 256]:
        for seed in [200, 201, 202]:
            for t in scaling_tasks:
                for a in scaling_archs:
                    queue.append(
                        {
                            "tier": "tier2_width_scaling",
                            "task": t,
                            "arch": a,
                            "d_model": d_m,
                            "max_train_len": 512,
                            "seed": seed,
                            "steps": steps,
                        }
                    )

    # TIER 3: Training Context Horizon Study (max_train_len in [64, 128, 256, 1024])
    for max_tr in [64, 128, 256, 1024]:
        for seed in [300, 301, 302]:
            for t in scaling_tasks:
                for a in ["GraphTree_Bidirectional", "GraphTree_BottomUp", "Flat_Transformer_RoPE"]:
                    queue.append(
                        {
                            "tier": "tier3_train_horizon",
                            "task": t,
                            "arch": a,
                            "d_model": 128,
                            "max_train_len": max_tr,
                            "seed": seed,
                            "steps": steps,
                        }
                    )

    # TIER 4: Deep Seed Expansion (Seeds 105..124 across all tasks & architectures)
    for seed_idx in range(base_seeds, 25):
        seed = 100 + seed_idx
        for t in TASKS:
            for a in ARCHS:
                queue.append(
                    {
                        "tier": "tier4_deep_seeds",
                        "task": t,
                        "arch": a,
                        "d_model": 128,
                        "max_train_len": 512,
                        "seed": seed,
                        "steps": steps,
                    }
                )

    return queue


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive GraphTree Evaluation Suite")
    parser.add_argument("--target_hours", type=float, default=18.0, help="Target runtime in hours")
    parser.add_argument("--steps", type=int, default=2200, help="Training steps per model run")
    parser.add_argument("--base_seeds", type=int, default=5, help="Initial seeds in Tier 1")
    parser.add_argument(
        "--max_eval_len",
        type=int,
        default=4194304,
        help="Maximum sequence length N to evaluate (default: 4,194,304)",
    )
    parser.add_argument("--out_dir", type=str, default="./paper_results", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "results_incremental.json")

    raw_records: List[Dict[str, Any]] = []
    completed_keys = set()
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw_records = json.load(f)
            for r in raw_records:
                k = (
                    r.get("tier", "tier1_main"),
                    r["task"],
                    r["arch"],
                    r.get("d_model", 128),
                    r.get("max_train_len", 512),
                    r["seed"],
                )
                completed_keys.add(k)
            print(f"Resumed {len(completed_keys)} completed runs from {json_path}")
        except Exception as e:
            print(f"Warning: could not load existing JSON ({e}), starting fresh.")

    suite = ComprehensiveLongContextSuite(device)
    eval_lengths = [length for length in DEFAULT_EVAL_LENGTHS if length <= args.max_eval_len]
    experiment_queue = build_experiment_queue(args.base_seeds, args.steps)

    print("=" * 100)
    print(
        f"Starting Comprehensive GraphTree Suite on {device} | "
        f"Target={args.target_hours:.1f}h | Total Planned Jobs={len(experiment_queue)}"
    )
    print(f"Incremental outputs saved to: {os.path.abspath(args.out_dir)}")
    print("=" * 100)

    suite_start = time.time()
    target_seconds = args.target_hours * 3600.0

    for idx, job in enumerate(experiment_queue, 1):
        elapsed_sec = time.time() - suite_start
        if elapsed_sec >= target_seconds:
            print(
                f"\n[Time Budget Reached] {elapsed_sec / 3600.0:.2f}h >= {args.target_hours:.2f}h. "
                f"Wrapping up cleanly."
            )
            break

        job_key = (
            job["tier"],
            job["task"],
            job["arch"],
            job["d_model"],
            job["max_train_len"],
            job["seed"],
        )
        if job_key in completed_keys:
            continue

        torch.manual_seed(job["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(job["seed"])

        hrs_elapsed = elapsed_sec / 3600.0
        hrs_rem = max(0.0, (target_seconds - elapsed_sec) / 3600.0)
        run_tag = (
            f"Job {idx}/{len(experiment_queue)} | {job['tier']} | {job['task']} | "
            f"{job['arch']} (d={job['d_model']}, Ntr<={job['max_train_len']}, s={job['seed']})"
        )
        print(f"\n--> [{hrs_elapsed:.2f}h elapsed, {hrs_rem:.2f}h left] {run_tag}")

        try:
            model = build_model(job["arch"], suite, d_model=job["d_model"]).to(device)
            is_flat = job["arch"].startswith("Flat_Transformer")
            lr = 1e-3 if is_flat else 2.5e-3

            train_sec = train_single_run(
                model=model,
                suite=suite,
                task_name=job["task"],
                steps=job["steps"],
                lr=lr,
                max_train_len=job["max_train_len"],
                run_tag=f"{job['task']}|{job['arch']}|s={job['seed']}",
            )

            eval_metrics = evaluate_single_run(
                model=model,
                suite=suite,
                task_name=job["task"],
                eval_lengths=eval_lengths,
                is_flat_transformer=is_flat,
            )

            summary_str = format_eval_line(eval_metrics, eval_lengths)
            print(f"    [Eval] {summary_str}")

            raw_records.append(
                {
                    "tier": job["tier"],
                    "task": job["task"],
                    "arch": job["arch"],
                    "d_model": job["d_model"],
                    "max_train_len": job["max_train_len"],
                    "seed": job["seed"],
                    "train_sec": round(train_sec, 2),
                    "eval": eval_metrics,
                }
            )
            completed_keys.add(job_key)
            export_all_reports(raw_records, TASKS, ARCHS, eval_lengths, args.out_dir)

        except Exception as e:
            print(f"    [ERROR in {run_tag}]: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_hrs = (time.time() - suite_start) / 3600.0
    print("\n" + "=" * 100)
    print(f"SUITE FINISHED in {total_hrs:.2f} hours ({len(raw_records)} total evaluations saved).")
    print(f"  - Main Summary CSV   : {os.path.join(args.out_dir, 'results_summary.csv')}")
    print(f"  - Scaling Laws CSV   : {os.path.join(args.out_dir, 'scaling_laws_summary.csv')}")
    print(f"  - LaTeX Paper Tables : {os.path.join(args.out_dir, 'paper_tables.tex')}")
    print(f"  - Raw Checkpoint JSON: {os.path.join(args.out_dir, 'results_incremental.json')}")
    print("=" * 100)


if __name__ == "__main__":
    main()