"""Sibling-GRPO: Tree-structured policy optimization for SID generation.

Implements the Sibling-GRPO objective from V-STAR (Sec 4.3):
- Sibling groups: candidates sharing a common parent prefix
- Node-level advantage: normalized mean reward per child node
- Sibling-GRPO loss: GRPO-style update on branching tokens with sibling advantages

Reference equations:
  Eq. 15: G(h) = {y in C(x) | y_{<l} = h}  (sibling group)
  Eq. 16: S(h) = {v | v in V, exists y in G(h) s.t. y_l = v}  (sibling node set)
  Eq. 17: R_bar(x; h, v) = mean reward of candidates routed through v
  Eq. 18: A_node(x; h, v) = (R_bar - mu_h) / (sigma_h + epsilon)
  Eq. 19: J_sib = E[ 1/|C| * sum_l sum_h sum_{v in S(h)} A_node * rho(v|x,h) ]
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def compute_sibling_advantages(
    candidate_sids: torch.Tensor,
    rewards: torch.Tensor,
    sid_length: int = 3,
    epsilon: float = 1e-4,
) -> Dict[Tuple, float]:
    """Compute sibling-relative node advantages for all (depth, parent, child) triples.

    Args:
        candidate_sids: [num_candidates, sid_length] tensor of SID token IDs.
        rewards: [num_candidates] tensor of per-candidate rewards.
        sid_length: number of SID levels (default 3).
        epsilon: small constant for numerical stability in normalization.

    Returns:
        Dictionary mapping (depth, parent_prefix_tuple, child_token) -> advantage float.
        Only entries with >= 2 sibling nodes are included.
    """
    sibling_advantages: Dict[Tuple, float] = {}
    num_candidates = candidate_sids.size(0)

    # Convert to lists for easier grouping
    sids_list = candidate_sids.tolist()
    rewards_list = rewards.tolist()

    for depth in range(1, sid_length + 1):  # l = 1, 2, 3
        # Group candidates by parent prefix (depth l-1 tokens)
        # parent_groups[parent_prefix][child_token] = list of rewards
        parent_groups: Dict[tuple, Dict[int, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for i in range(num_candidates):
            parent_prefix = tuple(sids_list[i][: depth - 1])  # h
            child_token = sids_list[i][depth - 1]  # v
            parent_groups[parent_prefix][child_token].append(rewards_list[i])

        for parent_prefix, children in parent_groups.items():
            if len(children) < 2:
                # Need at least 2 distinct sibling nodes for meaningful comparison
                continue

            # Eq. 17: Node-level score = mean reward of candidates routed through v
            node_scores: Dict[int, float] = {}
            for child_token, child_rewards in children.items():
                node_scores[child_token] = float(np.mean(child_rewards))

            # Eq. 18: Sibling-relative advantage
            scores = list(node_scores.values())
            mu_h = float(np.mean(scores))
            sigma_h = float(np.std(scores))

            for child_token, score in node_scores.items():
                adv = (score - mu_h) / (sigma_h + epsilon)
                sibling_advantages[(depth, parent_prefix, child_token)] = adv

    return sibling_advantages


def build_sibling_advantage_tensor(
    candidate_sids: torch.Tensor,
    sibling_advantages: Dict[Tuple, float],
    sid_length: int = 3,
) -> torch.Tensor:
    """Map per-candidate SID tokens to their sibling advantages at each depth.

    Args:
        candidate_sids: [num_candidates, sid_length] tensor of SID token IDs.
        sibling_advantages: output of compute_sibling_advantages.
        sid_length: number of SID levels.

    Returns:
        [num_candidates, sid_length] tensor of sibling advantages.
        Positions without a valid sibling advantage are set to 0.0.
    """
    num_candidates = candidate_sids.size(0)
    adv_tensor = torch.zeros(
        num_candidates, sid_length, device=candidate_sids.device, dtype=torch.float32
    )
    sids_list = candidate_sids.tolist()

    for i in range(num_candidates):
        for depth in range(1, sid_length + 1):
            parent_prefix = tuple(sids_list[i][: depth - 1])
            child_token = sids_list[i][depth - 1]
            key = (depth, parent_prefix, child_token)
            if key in sibling_advantages:
                adv_tensor[i, depth - 1] = sibling_advantages[key]

    return adv_tensor


def sibling_grpo_loss(
    per_token_logps: torch.Tensor,
    ref_per_token_logps: torch.Tensor,
    sibling_adv_tensor: torch.Tensor,
    completion_mask: torch.Tensor,
    sid_length: int = 3,
    beta: float = 1e-3,
) -> torch.Tensor:
    """Compute the Sibling-GRPO loss (Eq. 19).

    The loss applies GRPO-style updates on the first `sid_length` tokens of each
    completion, weighted by sibling-relative node advantages.

    Args:
        per_token_logps: [B, T] per-token log probabilities from current policy.
        ref_per_token_logps: [B, T] per-token log probabilities from reference policy.
        sibling_adv_tensor: [B, sid_length] sibling advantages per SID token.
        completion_mask: [B, T] binary mask for valid completion tokens.
        sid_length: number of SID levels (default 3).
        beta: KL penalty coefficient.

    Returns:
        Scalar loss tensor.
    """
    # Only operate on the first sid_length tokens (the SID branching tokens)
    T = per_token_logps.size(1)
    effective_len = min(sid_length, T)

    # Slice to SID tokens only
    sid_logps = per_token_logps[:, :effective_len]  # [B, L]
    sid_ref_logps = ref_per_token_logps[:, :effective_len]  # [B, L]
    sid_mask = completion_mask[:, :effective_len]  # [B, L]
    sid_advs = sibling_adv_tensor[:, :effective_len]  # [B, L]

    # Importance ratio (Eq. 19): rho = pi / pi_old
    # In log space: log(rho) = logps - ref_logps
    # GRPO-style: use exp(logps - logps.detach()) * advantage (same as standard GRPO)
    per_token_loss = torch.exp(sid_logps - sid_logps.detach()) * sid_advs

    # KL penalty on SID tokens
    per_token_kl = (
        torch.exp(sid_ref_logps - sid_logps)
        - (sid_ref_logps - sid_logps)
        - 1
    )
    # Clamp KL to prevent numerical explosion from dummy/fallback tokens
    per_token_kl = per_token_kl.clamp(max=100.0)

    per_token_loss = -(per_token_loss - beta * per_token_kl)

    # Mask and average
    # Only count positions where we have valid sibling advantages (non-zero)
    has_adv = (sid_advs != 0).float() * sid_mask
    if has_adv.sum() < 1:
        return torch.zeros((), device=per_token_logps.device, dtype=per_token_logps.dtype)

    loss = (per_token_loss * has_adv).sum() / has_adv.sum().clamp_min(1)
    return loss
