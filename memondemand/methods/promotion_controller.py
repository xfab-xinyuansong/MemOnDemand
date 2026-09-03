"""On-demand promotion controller used by V5 runners."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from memondemand.methods.dual_node import NODE_STATE_LIGHT, NODE_STATE_PROMOTED, DualNode
from memondemand.methods.state_log import EVENT_KEEP_LIGHT, EVENT_PROMOTE, StateLog


@dataclass
class PromotionDecision:
    node_id: str
    decision: str
    score: float
    rationale: str = ""
    query_id: str = ""
    query_idx: int = -1


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]{2,}", (text or "").lower()))


class PromotionController:
    """Budgeted promotion gate.

    The release implementation is deterministic and dependency-free. It scores
    candidates by lexical overlap between the query and each node's distilled
    plus detailed memory, then promotes the strongest still-light nodes while
    respecting the configured budget.
    """

    def __init__(
        self,
        hierarchy_dict: Dict[str, DualNode],
        embedder: Any,
        promotion_budget: int = 20,
        decay_window: int = 15,
        tau: float = 10.0,
        alias_high: str = "",
        alias_low: str = "",
        max_candidates_per_decision: int = 6,
        deterministic_only: bool = True,
    ):
        self.hierarchy = hierarchy_dict
        self.embedder = embedder
        self.promotion_budget = promotion_budget
        self.decay_window = decay_window
        self.tau = tau
        self.alias_high = alias_high
        self.alias_low = alias_low
        self.max_candidates_per_decision = max_candidates_per_decision
        self.deterministic_only = deterministic_only

    def _n_promoted(self) -> int:
        return sum(1 for node in self.hierarchy.values() if node.state == NODE_STATE_PROMOTED)

    def _score(self, query: str, node: DualNode) -> float:
        q = _tokens(query)
        if not q:
            return 0.0
        body = _tokens(" ".join([node.key_facts, node.distilled_text, node.detailed_text]))
        if not body:
            return 0.0
        overlap = len(q & body) / math.sqrt(max(1, len(q)) * max(1, len(body)))
        return float(min(1.0, overlap))

    def decide(
        self,
        query: str,
        candidate_node_ids: Iterable[str],
        query_idx: int,
        query_id: str = "",
        alias: str = "",
        ledger: Optional[Any] = None,
    ) -> List[PromotionDecision]:
        del alias, ledger
        candidates = [nid for nid in candidate_node_ids if nid in self.hierarchy]
        scored = [
            (nid, self._score(query, self.hierarchy[nid]))
            for nid in candidates[: self.max_candidates_per_decision]
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        remaining = max(0, self.promotion_budget - self._n_promoted())

        decisions: List[PromotionDecision] = []
        for rank, (nid, score) in enumerate(scored):
            node = self.hierarchy[nid]
            should_promote = (
                remaining > 0
                and node.state == NODE_STATE_LIGHT
                and (score > 0.0 or rank == 0)
            )
            if should_promote:
                remaining -= 1
                decisions.append(
                    PromotionDecision(
                        node_id=nid,
                        decision=EVENT_PROMOTE,
                        score=score,
                        rationale="highest lexical overlap under promotion budget",
                        query_id=query_id,
                        query_idx=query_idx,
                    )
                )
            else:
                decisions.append(
                    PromotionDecision(
                        node_id=nid,
                        decision=EVENT_KEEP_LIGHT,
                        score=score,
                        rationale="budget exhausted, already promoted, or weak overlap",
                        query_id=query_id,
                        query_idx=query_idx,
                    )
                )
        return decisions

    def apply_decisions(
        self,
        decisions: Iterable[PromotionDecision],
        query_idx: int,
        state_log: Optional[StateLog] = None,
    ) -> Dict[str, int]:
        counts = {EVENT_PROMOTE: 0, EVENT_KEEP_LIGHT: 0}
        for decision in decisions:
            node = self.hierarchy.get(decision.node_id)
            if node is None:
                continue
            if decision.decision == EVENT_PROMOTE:
                node.state = NODE_STATE_PROMOTED
                node.promotion_score = decision.score
                node.last_promoted_query_idx = query_idx
                node.promotion_count += 1
                counts[EVENT_PROMOTE] += 1
            else:
                counts[EVENT_KEEP_LIGHT] += 1
            if state_log is not None:
                state_log.record(
                    decision.decision,
                    decision.node_id,
                    query_idx,
                    query_id=decision.query_id,
                    score=decision.score,
                    reason=decision.rationale,
                )
        return counts

    def mark_detail_used(
        self,
        node_id: str,
        query_idx: int,
        query_id: str = "",
        state_log: Optional[StateLog] = None,
    ) -> None:
        node = self.hierarchy.get(node_id)
        if node is None:
            return
        node.last_used_query_idx = query_idx
        node.detail_use_count += 1
        if state_log is not None:
            state_log.record(
                "DETAIL_USED",
                node_id,
                query_idx,
                query_id=query_id,
                score=node.promotion_score,
                reason="included in answer context",
            )
