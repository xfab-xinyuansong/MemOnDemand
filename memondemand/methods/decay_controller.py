"""Decay controller for budgeted promoted memory."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from memondemand.methods.dual_node import NODE_STATE_LIGHT, NODE_STATE_PROMOTED, DualNode
from memondemand.methods.state_log import EVENT_DEMOTE, StateLog


class DecayController:
    def __init__(self, promotion_budget: int = 20, decay_window: int = 15, tau: float = 10.0):
        self.promotion_budget = promotion_budget
        self.decay_window = decay_window
        self.tau = tau

    def select_for_demotion(
        self, hierarchy: Dict[str, DualNode], query_idx: int
    ) -> Tuple[List[str], Dict[str, str], Dict[str, float]]:
        promoted = [
            node
            for node in hierarchy.values()
            if node.state == NODE_STATE_PROMOTED
        ]
        reasons: Dict[str, str] = {}
        keep_scores: Dict[str, float] = {}
        demote: List[str] = []

        for node in promoted:
            last_touch = max(node.last_used_query_idx, node.last_promoted_query_idx)
            age = query_idx - last_touch if last_touch >= 0 else query_idx + 1
            score = max(0.0, float(node.promotion_score) - age / max(self.tau, 1e-6))
            keep_scores[node.node_id] = score
            if age >= self.decay_window:
                demote.append(node.node_id)
                reasons[node.node_id] = f"stale for {age} queries"

        if len(promoted) - len(demote) > self.promotion_budget:
            survivors = [n for n in promoted if n.node_id not in set(demote)]
            survivors.sort(key=lambda n: keep_scores.get(n.node_id, 0.0))
            overflow = len(survivors) - self.promotion_budget
            for node in survivors[:overflow]:
                demote.append(node.node_id)
                reasons[node.node_id] = "promotion budget overflow"

        return demote, reasons, keep_scores

    def apply_demotions(
        self,
        hierarchy: Dict[str, DualNode],
        demote_ids: Iterable[str],
        query_idx: int,
        query_id: str = "",
        state_log: Optional[StateLog] = None,
        keep_scores: Optional[Dict[str, float]] = None,
        reasons: Optional[Dict[str, str]] = None,
    ) -> None:
        keep_scores = keep_scores or {}
        reasons = reasons or {}
        for node_id in demote_ids:
            node = hierarchy.get(node_id)
            if node is None:
                continue
            node.state = NODE_STATE_LIGHT
            if state_log is not None:
                state_log.record(
                    EVENT_DEMOTE,
                    node_id,
                    query_idx,
                    query_id=query_id,
                    score=keep_scores.get(node_id, 0.0),
                    reason=reasons.get(node_id, "decay"),
                )
