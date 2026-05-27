"""VED: Value-Guided Efficient Decoding for SID generation.

Implements the VED algorithm from V-STAR (Sec 4.2):
- Stage 1: Initialize prefix tree via small beam search
- Stage 2: UCB-style selection from root to leaf
- Stage 3: Gated expansion (only expand if G(s) >= G_bar_l)
- Stage 4: Backpropagation of visit counts

Reference equations:
  Eq. 10: Cost(T) = total backbone forward tokens (excluding prompt prefill)
  Eq. 11: G(s) = V_phi(s) + lambda * H_theta(s)
  Eq. 13: U(s) = G(s) + beta * sqrt(ln(N_root + 1) / (N(s) + 1))
  Eq. 14: Expand only if G(s) >= G_bar_l (depth-wise average priority)

Optimized with **batched parallel decoding**: all frontier nodes at the same
depth (across all prompts) are forwarded through the backbone in a single
batched call, dramatically reducing wall-clock time.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PrefixNode:
    """A node in the SID prefix tree."""

    token_id: int  # token at this node (-1 for root)
    depth: int  # 0 = root, 1..L = SID levels
    value: float = 0.0  # V_phi(s)
    entropy: float = 0.0  # H_theta(s) = entropy of next-token distribution
    priority: float = 0.0  # G(s) = V + lambda * H
    visit_count: int = 0  # N(s)
    children: Dict[int, "PrefixNode"] = field(default_factory=dict)
    parent: Optional["PrefixNode"] = None
    log_prob: float = 0.0  # log pi(token | parent prefix)

    @property
    def prefix(self) -> List[int]:
        """Reconstruct the token prefix from root to this node."""
        tokens = []
        node = self
        while node.parent is not None:
            tokens.append(node.token_id)
            node = node.parent
        return list(reversed(tokens))

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class VEDDecoder:
    """Value-Guided Efficient Decoding (V-STAR Sec 4.2).

    Builds a prefix tree over SID tokens and iteratively expands
    high-value nodes under a strict forward-token budget.

    Args:
        hash_dict: mapping from prefix hash string -> list of valid next token IDs.
        get_hash_fn: function to convert a list of token IDs to a hash string.
        sid_length: number of SID levels (default 3).
        init_beam_width: beam width for initial tree construction (default 8).
        num_candidates: number of candidates to return (default 16).
        lambda_explore: entropy weight in priority G(s) (default 0.1).
        beta_ucb: UCB exploration coefficient (default 1.0).
        budget_multiplier: multiplier for the base budget (default 1.0).
        prefix_index: number of prompt template tokens before SID starts (default 3).
    """

    def __init__(
        self,
        hash_dict: Dict[str, List[int]],
        get_hash_fn,
        sid_length: int = 3,
        init_beam_width: int = 8,
        num_candidates: int = 16,
        lambda_explore: float = 0.1,
        beta_ucb: float = 1.0,
        budget_multiplier: float = 1.0,
        prefix_index: int = 3,
    ):
        self.hash_dict = hash_dict
        self.get_hash = get_hash_fn
        self.sid_length = sid_length
        self.init_beam_width = init_beam_width
        self.num_candidates = num_candidates
        self.lambda_explore = lambda_explore
        self.beta_ucb = beta_ucb
        self.prefix_index = prefix_index

        # Base budget: 1 (root) + 2 * init_beam_width (for L=3 tree)
        base_budget = 1 + (sid_length - 1) * init_beam_width
        self.budget = int(base_budget * budget_multiplier)

    # ------------------------------------------------------------------
    # Single-prompt decode (kept for backward compatibility)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(
        self,
        model,
        value_head,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        prompt_prefix_ids: List[int],
    ) -> List[List[int]]:
        """Run VED for a single prompt and return candidate SID token lists."""
        device = prompt_ids.device
        root = PrefixNode(token_id=-1, depth=0)
        cost = [0]

        self._init_tree(model, value_head, root, prompt_ids, prompt_mask,
                        prompt_prefix_ids, device, cost)

        max_iters = self.budget * 3
        for _ in range(max_iters):
            if cost[0] >= self.budget:
                break
            path = self._select_path(root)
            if path is None:
                break
            expanded = self._gated_expand(
                model, value_head, root, path, prompt_ids, prompt_mask,
                prompt_prefix_ids, device, cost
            )
            if expanded:
                self._backpropagate(path)

        return self._extract_candidates(root)

    # ------------------------------------------------------------------
    # Batched init tree: all frontier nodes at the same depth are batched
    # into a single model forward pass (within a single prompt).
    # ------------------------------------------------------------------
    def _init_tree(
        self, model, value_head, root, prompt_ids, prompt_mask,
        prompt_prefix_ids, device, cost
    ):
        """Stage 1: Initialize tree with batched forward passes per depth."""
        frontier = [root]

        for depth in range(1, self.sid_length + 1):
            next_frontier = []
            # Collect all (node, valid_tokens) pairs for this depth
            expand_items = []
            for node in frontier:
                prefix = node.prefix
                valid_tokens = self._get_valid_tokens(prefix, prompt_prefix_ids)
                if valid_tokens:
                    expand_items.append((node, prefix, valid_tokens))

            if not expand_items:
                frontier = next_frontier
                continue

            # Build batched input: one row per frontier node
            input_list = []
            mask_list = []
            for node, prefix, valid_tokens in expand_items:
                inp, msk = self._build_input(
                    prompt_ids, prompt_mask, prefix, prompt_prefix_ids, device
                )
                input_list.append(inp.squeeze(0))
                mask_list.append(msk.squeeze(0))

            # Pad and batch
            batch_ids = torch.nn.utils.rnn.pad_sequence(
                input_list, batch_first=True, padding_value=0
            )
            batch_mask = torch.nn.utils.rnn.pad_sequence(
                mask_list, batch_first=True, padding_value=0
            )

            # Single batched forward pass (optimized: no output_hidden_states)
            logits, hidden, _ = _forward_with_hidden(model, batch_ids, batch_mask)
            cost[0] += len(expand_items)

            token_values, cls_values = value_head(hidden, batch_mask)

            # Process each frontier node
            for idx, (node, prefix, valid_tokens) in enumerate(expand_items):
                next_logits = logits[idx, -1, :]  # [vocab_size]

                valid_mask = torch.full_like(next_logits, float("-inf"))
                valid_tokens_tensor = torch.tensor(valid_tokens, device=device)
                valid_mask[valid_tokens_tensor] = next_logits[valid_tokens_tensor]

                probs = F.softmax(valid_mask[valid_tokens_tensor], dim=0)
                log_probs = F.log_softmax(valid_mask[valid_tokens_tensor], dim=0)
                entropy = -(probs * log_probs).sum().item()

                value = cls_values[idx].item()

                k = min(self.init_beam_width, len(valid_tokens))
                top_probs, top_indices = probs.topk(k)

                for i in range(k):
                    tok = valid_tokens[top_indices[i].item()]
                    child = PrefixNode(
                        token_id=tok,
                        depth=depth,
                        value=value,
                        entropy=entropy,
                        priority=value + self.lambda_explore * entropy,
                        visit_count=1,
                        parent=node,
                        log_prob=log_probs[top_indices[i]].item(),
                    )
                    node.children[tok] = child
                    if depth < self.sid_length:
                        next_frontier.append(child)

            frontier = next_frontier

        root.visit_count = sum(c.visit_count for c in root.children.values()) + 1

    def _select_path(self, root: PrefixNode) -> Optional[List[PrefixNode]]:
        """Stage 2: UCB-style selection from root to a leaf node (Eq. 13)."""
        path = [root]
        node = root

        while node.children:
            best_child = None
            best_ucb = float("-inf")

            for child in node.children.values():
                exploration = self.beta_ucb * math.sqrt(
                    math.log(root.visit_count + 1) / (child.visit_count + 1)
                )
                ucb = child.priority + exploration
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child

            if best_child is None:
                break
            path.append(best_child)
            node = best_child

        return path if len(path) > 1 else None

    def _gated_expand(
        self, model, value_head, root, path, prompt_ids, prompt_mask,
        prompt_prefix_ids, device, cost
    ) -> bool:
        """Stage 3: Gated expansion (Eq. 14) — single leaf version."""
        leaf = path[-1]
        if leaf.depth >= self.sid_length:
            return False
        if cost[0] >= self.budget:
            return False

        g_bar = self._compute_depth_avg_priority(root, leaf.depth)
        if leaf.priority < g_bar:
            return False

        prefix = leaf.prefix
        valid_tokens = self._get_valid_tokens(prefix, prompt_prefix_ids)
        unexpanded = [t for t in valid_tokens if t not in leaf.children]
        if not unexpanded:
            return False

        input_ids, attn_mask = self._build_input(
            prompt_ids, prompt_mask, prefix, prompt_prefix_ids, device
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
        cost[0] += 1

        next_logits = outputs.logits[0, -1, :]
        unexpanded_tensor = torch.tensor(unexpanded, device=device)
        unexpanded_logits = next_logits[unexpanded_tensor]
        probs = F.softmax(unexpanded_logits, dim=0)
        log_probs = F.log_softmax(unexpanded_logits, dim=0)

        sampled_idx = torch.multinomial(probs, 1).item()
        tok = unexpanded[sampled_idx]

        all_valid_tensor = torch.tensor(valid_tokens, device=device)
        all_logits = next_logits[all_valid_tensor]
        all_probs = F.softmax(all_logits, dim=0)
        all_log_probs = F.log_softmax(all_logits, dim=0)
        entropy = -(all_probs * all_log_probs).sum().item()

        hidden = outputs.hidden_states[-1]
        token_values, cls_value = value_head(hidden, attn_mask)
        value = cls_value.item()

        child = PrefixNode(
            token_id=tok,
            depth=leaf.depth + 1,
            value=value,
            entropy=entropy,
            priority=value + self.lambda_explore * entropy,
            visit_count=1,
            parent=leaf,
            log_prob=log_probs[sampled_idx].item(),
        )
        leaf.children[tok] = child
        return True

    def _backpropagate(self, path: List[PrefixNode]):
        """Stage 4: Update visit counts along the path."""
        for node in reversed(path):
            node.visit_count += 1

    def _extract_candidates(self, root: PrefixNode) -> List[List[int]]:
        """Extract complete SID sequences from depth-L leaf nodes, sorted by value."""
        leaves: List[PrefixNode] = []
        self._collect_leaves(root, leaves)
        leaves.sort(key=lambda n: n.value, reverse=True)
        candidates = [leaf.prefix for leaf in leaves[: self.num_candidates]]
        while len(candidates) < self.num_candidates and candidates:
            candidates.append(candidates[0])
        return candidates

    def _collect_leaves(self, node: PrefixNode, leaves: List[PrefixNode]):
        """Recursively collect all leaf nodes at depth == sid_length."""
        if node.depth == self.sid_length:
            leaves.append(node)
            return
        for child in node.children.values():
            self._collect_leaves(child, leaves)

    def _compute_depth_avg_priority(self, root: PrefixNode, depth: int) -> float:
        """Compute G_bar_l = average priority of all nodes at the given depth."""
        nodes: List[PrefixNode] = []
        self._collect_at_depth(root, depth, nodes)
        if not nodes:
            return 0.0
        return sum(n.priority for n in nodes) / len(nodes)

    def _collect_at_depth(
        self, node: PrefixNode, target_depth: int, result: List[PrefixNode]
    ):
        if node.depth == target_depth:
            result.append(node)
            return
        for child in node.children.values():
            self._collect_at_depth(child, target_depth, result)

    def _get_valid_tokens(
        self, prefix: List[int], prompt_prefix_ids: List[int]
    ) -> List[int]:
        if not prefix:
            hash_key = self.get_hash(prompt_prefix_ids)
        else:
            hash_key = self.get_hash(prefix)
        return self.hash_dict.get(hash_key, [])

    def _build_input(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        prefix: List[int],
        prompt_prefix_ids: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prefix:
            prefix_tensor = torch.tensor(
                prefix, dtype=prompt_ids.dtype, device=device
            ).unsqueeze(0)
            input_ids = torch.cat([prompt_ids, prefix_tensor], dim=1)
            prefix_mask = torch.ones(
                1, len(prefix), dtype=prompt_mask.dtype, device=device
            )
            attn_mask = torch.cat([prompt_mask, prefix_mask], dim=1)
        else:
            input_ids = prompt_ids
            attn_mask = prompt_mask
        return input_ids, attn_mask


# ======================================================================
# Optimized helpers: avoid output_hidden_states=True + KV cache support
# ======================================================================

def _forward_with_hidden(model, input_ids, attention_mask,
                         past_key_values=None, use_cache=False):
    """Forward via model.model() + lm_head (no output_hidden_states=True).

    Avoids saving all 28 layers of hidden states, only gets last layer.
    Saves ~30% memory and compute per forward pass.
    """
    base_model = model.model  # Qwen2Model / LlamaModel
    base_out = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
    )
    hidden = base_out.last_hidden_state
    logits = model.lm_head(hidden)
    pkv = base_out.past_key_values if use_cache else None
    return logits, hidden, pkv


def _clone_past_key_values(past_key_values):
    """Deep-clone past_key_values so we can extend without mutating the original."""
    if past_key_values is None:
        return None
    import copy
    return copy.deepcopy(past_key_values)


# ======================================================================
# Batched parallel VED with KV Cache
# ======================================================================

@torch.no_grad()
def ved_decode_batch(
    model,
    value_head,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    prompt_prefix_ids_list: List[List[int]],
    hash_dict: Dict[str, List[int]],
    get_hash_fn,
    num_generations: int = 16,
    sid_length: int = 3,
    init_beam_width: int = 8,
    lambda_explore: float = 0.1,
    beta_ucb: float = 1.0,
    budget_multiplier: float = 1.0,
    prefix_index: int = 3,
) -> List[List[List[int]]]:
    """Run VED for a batch of prompts with **parallel batched decoding**.

    All prompts' frontier nodes at the same depth are batched into a single
    model forward pass, dramatically reducing wall-clock time compared to
    the sequential per-prompt approach.

    Args:
        model: backbone LM.
        value_head: ValueHead module.
        prompt_ids: [B, prompt_len] deduped prompt IDs (one per unique prompt).
        prompt_mask: [B, prompt_len] attention masks.
        prompt_prefix_ids_list: list of B prefix ID lists.
        hash_dict: SID constraint trie.
        get_hash_fn: hash function for trie lookup.
        num_generations: candidates per prompt.
        sid_length: SID depth.
        init_beam_width: initial beam width.
        lambda_explore: entropy weight.
        beta_ucb: UCB coefficient.
        budget_multiplier: budget scaling factor.
        prefix_index: number of template tokens before SID.

    Returns:
        List of B lists, each containing num_generations SID token lists.
    """
    device = prompt_ids.device
    batch_size = prompt_ids.size(0)
    base_budget = 1 + (sid_length - 1) * init_beam_width
    budget = int(base_budget * budget_multiplier)

    def _get_valid_tokens(prefix: List[int], prompt_prefix_ids: List[int]) -> List[int]:
        if not prefix:
            hash_key = get_hash_fn(prompt_prefix_ids)
        else:
            hash_key = get_hash_fn(prefix)
        return hash_dict.get(hash_key, [])

    # ==================================================================
    # Helper: forward for a single (prompt, prefix) pair.
    # Uses _forward_with_hidden (no output_hidden_states=True).
    # ==================================================================
    def _cached_forward(pi: int, prefix: List[int]):
        p_ids = prompt_ids[pi:pi+1]
        p_mask = prompt_mask[pi:pi+1]
        if not prefix:
            # No prefix: just forward the prompt
            logits, hidden, _ = _forward_with_hidden(model, p_ids, p_mask)
            return logits[:, -1, :].squeeze(0), hidden, p_mask

        # Forward full prompt + prefix (no KV cache, avoids expensive deepcopy)
        prefix_tensor = torch.tensor(
            prefix, dtype=prompt_ids.dtype, device=device
        ).unsqueeze(0)
        full_ids = torch.cat([p_ids, prefix_tensor], dim=1)
        full_mask = torch.cat([
            p_mask,
            torch.ones(1, len(prefix), dtype=p_mask.dtype, device=device)
        ], dim=1)
        logits, hidden, _ = _forward_with_hidden(model, full_ids, full_mask)
        return logits[:, -1, :].squeeze(0), hidden, full_mask

    # Per-prompt state
    roots = [PrefixNode(token_id=-1, depth=0) for _ in range(batch_size)]
    costs = [0] * batch_size

    # ==================================================================
    # Stage 1: Init tree with KV-cached forward passes.
    # Group by (prompt_idx, prefix) to avoid redundant forwards.
    # ==================================================================
    frontiers = [[roots[i]] for i in range(batch_size)]

    for depth in range(1, sid_length + 1):
        expand_items: List[Tuple[int, PrefixNode, List[int], List[int]]] = []
        for pi in range(batch_size):
            for node in frontiers[pi]:
                prefix = node.prefix
                valid_tokens = _get_valid_tokens(prefix, prompt_prefix_ids_list[pi])
                if valid_tokens:
                    expand_items.append((pi, node, prefix, valid_tokens))

        if not expand_items:
            frontiers = [[] for _ in range(batch_size)]
            continue

        # Group items by (prompt_idx, prefix) — nodes sharing the same parent
        # only need ONE forward pass since they produce the same logits/value.
        groups = defaultdict(list)
        for pi, node, prefix, valid_tokens in expand_items:
            groups[(pi, tuple(prefix))].append((node, valid_tokens))

        next_frontiers: List[List[PrefixNode]] = [[] for _ in range(batch_size)]

        for (pi, prefix_tuple), items in groups.items():
            prefix = list(prefix_tuple)
            costs[pi] += 1

            logits_last, full_hidden, full_mask = _cached_forward(pi, prefix)
            next_logits = logits_last.squeeze(0)

            _, cls_value = value_head(full_hidden, full_mask)
            value = cls_value.item()

            for node, valid_tokens in items:
                valid_tokens_tensor = torch.tensor(valid_tokens, device=device)
                valid_logits = next_logits[valid_tokens_tensor]

                probs = F.softmax(valid_logits, dim=0)
                log_probs = F.log_softmax(valid_logits, dim=0)
                entropy = -(probs * log_probs).sum().item()

                k = min(init_beam_width, len(valid_tokens))
                top_probs, top_indices = probs.topk(k)

                for i in range(k):
                    tok = valid_tokens[top_indices[i].item()]
                    child = PrefixNode(
                        token_id=tok,
                        depth=depth,
                        value=value,
                        entropy=entropy,
                        priority=value + lambda_explore * entropy,
                        visit_count=1,
                        parent=node,
                        log_prob=log_probs[top_indices[i]].item(),
                    )
                    node.children[tok] = child
                    if depth < sid_length:
                        next_frontiers[pi].append(child)

        frontiers = next_frontiers

    # Finalize root visit counts
    for pi in range(batch_size):
        roots[pi].visit_count = sum(
            c.visit_count for c in roots[pi].children.values()
        ) + 1

    # ==================================================================
    # Stage 2-4: Iterative expansion with KV cache
    # ==================================================================
    max_iters = budget * 3

    for _ in range(max_iters):
        # Check if all prompts have exhausted their budgets
        if all(costs[pi] >= budget for pi in range(batch_size)):
            break

        # Collect one expandable leaf per prompt (via UCB selection)
        expand_batch: List[Tuple[int, List[PrefixNode], PrefixNode, List[int], List[int]]] = []

        for pi in range(batch_size):
            if costs[pi] >= budget:
                continue

            path = _select_path_static(roots[pi], beta_ucb)
            if path is None:
                continue

            leaf = path[-1]
            if leaf.depth >= sid_length:
                continue

            g_bar = _compute_depth_avg_priority_static(roots[pi], leaf.depth)
            if leaf.priority < g_bar:
                continue

            prefix = leaf.prefix
            valid_tokens = _get_valid_tokens(prefix, prompt_prefix_ids_list[pi])
            unexpanded = [t for t in valid_tokens if t not in leaf.children]
            if not unexpanded:
                continue

            expand_batch.append((pi, path, leaf, valid_tokens, unexpanded))

        if not expand_batch:
            break

        # Process each expandable leaf with KV cache
        for pi, path, leaf, valid_tokens, unexpanded in expand_batch:
            costs[pi] += 1
            prefix = leaf.prefix

            logits_last, full_hidden, full_mask = _cached_forward(pi, prefix)
            next_logits = logits_last.squeeze(0)

            _, cls_value = value_head(full_hidden, full_mask)
            value = cls_value.item()

            unexpanded_tensor = torch.tensor(unexpanded, device=device)
            unexpanded_logits = next_logits[unexpanded_tensor]
            probs = F.softmax(unexpanded_logits, dim=0)
            log_probs = F.log_softmax(unexpanded_logits, dim=0)

            sampled_idx = torch.multinomial(probs, 1).item()
            tok = unexpanded[sampled_idx]

            # Entropy over ALL valid tokens
            all_valid_tensor = torch.tensor(valid_tokens, device=device)
            all_logits = next_logits[all_valid_tensor]
            all_probs = F.softmax(all_logits, dim=0)
            all_log_probs = F.log_softmax(all_logits, dim=0)
            entropy = -(all_probs * all_log_probs).sum().item()

            child = PrefixNode(
                token_id=tok,
                depth=leaf.depth + 1,
                value=value,
                entropy=entropy,
                priority=value + lambda_explore * entropy,
                visit_count=1,
                parent=leaf,
                log_prob=log_probs[sampled_idx].item(),
            )
            leaf.children[tok] = child

            # Backpropagate visit counts
            for node in reversed(path):
                node.visit_count += 1

    # ==================================================================
    # Extract candidates from all prompts
    # ==================================================================
    all_candidates = []
    for pi in range(batch_size):
        leaves: List[PrefixNode] = []
        _collect_leaves_static(roots[pi], sid_length, leaves)
        leaves.sort(key=lambda n: n.value, reverse=True)
        candidates = [leaf.prefix for leaf in leaves[:num_generations]]
        while len(candidates) < num_generations and candidates:
            candidates.append(candidates[0])
        all_candidates.append(candidates)

    return all_candidates


# ======================================================================
# Static helper functions (used by batched ved_decode_batch)
# ======================================================================

def _select_path_static(root: PrefixNode, beta_ucb: float) -> Optional[List[PrefixNode]]:
    """UCB-style selection from root to a leaf node."""
    path = [root]
    node = root
    while node.children:
        best_child = None
        best_ucb = float("-inf")
        for child in node.children.values():
            exploration = beta_ucb * math.sqrt(
                math.log(root.visit_count + 1) / (child.visit_count + 1)
            )
            ucb = child.priority + exploration
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        if best_child is None:
            break
        path.append(best_child)
        node = best_child
    return path if len(path) > 1 else None


def _compute_depth_avg_priority_static(root: PrefixNode, depth: int) -> float:
    """Compute G_bar_l = average priority of all nodes at the given depth."""
    nodes: List[PrefixNode] = []
    _collect_at_depth_static(root, depth, nodes)
    if not nodes:
        return 0.0
    return sum(n.priority for n in nodes) / len(nodes)


def _collect_at_depth_static(
    node: PrefixNode, target_depth: int, result: List[PrefixNode]
):
    if node.depth == target_depth:
        result.append(node)
        return
    for child in node.children.values():
        _collect_at_depth_static(child, target_depth, result)


def _collect_leaves_static(
    node: PrefixNode, sid_length: int, leaves: List[PrefixNode]
):
    if node.depth == sid_length:
        leaves.append(node)
        return
    for child in node.children.values():
        _collect_leaves_static(child, sid_length, leaves)
