"""Per-phase, per-model token accounting.

Per the MemOnDemand token-accounting design, production runs report separately:
  - hierarchy_build_input/output
  - distilled_representation
  - detailed_representation
  - distilled_retrieval/navigation
  - promotion_decision
  - promoted_detailed_context
  - final_answer
  - per-query and per-correct-answer aggregates

This ledger keeps a single in-memory structure for the lifetime of a run, and
writes it out at end as JSON. It is thread-safe (one lock around mutations).

It also estimates API cost using a simple per-model price table (configurable).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Price table (USD per 1M tokens). Approximate as of 2026; values can be
# overridden via TokenLedger(prices={...}).
# ---------------------------------------------------------------------------

DEFAULT_PRICES = {
    "gpt_5_4_mini": {"input": 0.15, "output": 0.60},
    "gpt_5_4": {"input": 1.25, "output": 10.00},  # openai.gpt-5.4 via Bedrock-mantle (reference pricing 2026-06-09)
    "gpt_4o_mini": {"input": 0.15, "output": 0.60},
}


# ---------------------------------------------------------------------------
# Categories — the MemOnDemand token-accounting design
# ---------------------------------------------------------------------------

PHASE_HIERARCHY_BUILD = "hierarchy_build"
PHASE_DISTILLED_GEN = "distilled_text_gen"
PHASE_DETAILED_GEN = "detailed_text_gen"  # in current runners, detailed=raw L0, no LLM
PHASE_RETRIEVAL = "retrieval"
PHASE_PROMOTION_DECISION = "promotion_decision"
PHASE_PROMOTED_CONTEXT = "promoted_context"
PHASE_FINAL_ANSWER = "final_answer"
PHASE_JUDGE = "judge"

ALL_PHASES = (
    PHASE_HIERARCHY_BUILD, PHASE_DISTILLED_GEN, PHASE_DETAILED_GEN,
    PHASE_RETRIEVAL, PHASE_PROMOTION_DECISION, PHASE_PROMOTED_CONTEXT,
    PHASE_FINAL_ANSWER, PHASE_JUDGE,
)


@dataclass
class CallRecord:
    """One LLM API call."""
    phase: str
    model_alias: str
    input_tokens: int
    output_tokens: int
    wall_seconds: float
    cost_usd: float
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)


class TokenLedger:
    """In-memory ledger; flush to JSON at end."""

    def __init__(self, prices: Optional[Dict[str, Dict[str, float]]] = None,
                 run_id: str = "", method: str = "",
                 alias_status: str = "",
                 alias_chosen_at: str = "",
                 alias_chosen_by: str = ""):
        self.prices = prices or DEFAULT_PRICES
        # Reentrant lock — grand_total() is called from within export() which
        # already holds the lock.
        self._lock = threading.RLock()
        self._records: List[CallRecord] = []
        self._totals_by_phase: Dict[str, Dict[str, int]] = {}
        self._totals_by_model: Dict[str, Dict[str, int]] = {}
        self.run_id = run_id
        self.method = method
        self.t_start = time.time()
        # PROVISIONAL alias tracking (release-tracking metadata)
        self.alias_status = alias_status
        self.alias_chosen_at = alias_chosen_at
        self.alias_chosen_by = alias_chosen_by

    def _compute_cost(self, model_alias: str, in_t: int, out_t: int) -> float:
        prices = self.prices.get(model_alias) or self.prices.get("gpt_5_4_mini", {"input": 0.15, "output": 0.60})
        return in_t / 1e6 * prices["input"] + out_t / 1e6 * prices["output"]

    def record(self, phase: str, model_alias: str, input_tokens: int,
               output_tokens: int, wall_seconds: float, **extra) -> None:
        if phase not in ALL_PHASES:
            # Allow custom phases but warn (we don't raise — flexibility for ablations).
            pass
        cost = self._compute_cost(model_alias, input_tokens, output_tokens)
        rec = CallRecord(
            phase=phase, model_alias=model_alias,
            input_tokens=input_tokens, output_tokens=output_tokens,
            wall_seconds=wall_seconds, cost_usd=cost, extra=extra,
        )
        with self._lock:
            self._records.append(rec)
            t_phase = self._totals_by_phase.setdefault(phase, {"input": 0, "output": 0, "calls": 0, "wall": 0.0, "cost": 0.0})
            t_phase["input"] += input_tokens
            t_phase["output"] += output_tokens
            t_phase["calls"] += 1
            t_phase["wall"] += wall_seconds
            t_phase["cost"] += cost
            t_model = self._totals_by_model.setdefault(model_alias, {"input": 0, "output": 0, "calls": 0, "cost": 0.0})
            t_model["input"] += input_tokens
            t_model["output"] += output_tokens
            t_model["calls"] += 1
            t_model["cost"] += cost

    def grand_total(self) -> Dict[str, Any]:
        with self._lock:
            total_in = sum(r.input_tokens for r in self._records)
            total_out = sum(r.output_tokens for r in self._records)
            total_cost = sum(r.cost_usd for r in self._records)
            n_calls = len(self._records)
        return {
            "calls": n_calls,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "cost_usd": round(total_cost, 4),
            "wall_seconds_elapsed": round(time.time() - self.t_start, 1),
        }

    def export(self, path: str, include_raw: bool = False) -> None:
        with self._lock:
            doc = {
                "ledger_version": "v4.0.0",
                "run_id": self.run_id,
                "method": self.method,
                "alias_status": self.alias_status,
                "alias_chosen_at": self.alias_chosen_at,
                "alias_chosen_by": self.alias_chosen_by,
                "totals": self.grand_total(),
                "by_phase": {k: {**v, "cost": round(v["cost"], 6), "wall": round(v["wall"], 2)}
                             for k, v in self._totals_by_phase.items()},
                "by_model": {k: {**v, "cost": round(v["cost"], 6)}
                             for k, v in self._totals_by_model.items()},
                "prices_table": self.prices,
            }
            if include_raw:
                doc["records"] = [
                    {
                        "phase": r.phase, "model_alias": r.model_alias,
                        "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                        "wall_seconds": round(r.wall_seconds, 3),
                        "cost_usd": round(r.cost_usd, 6),
                        "timestamp": r.timestamp, "extra": r.extra,
                    }
                    for r in self._records
                ]
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)


def _self_test() -> int:
    led = TokenLedger(run_id="t1", method="MemOnDemand", alias_status="PROVISIONAL")
    led.record("distilled_text_gen", "gpt_5_4_mini", 100, 30, 0.5)
    led.record("distilled_text_gen", "gpt_5_4_mini", 120, 28, 0.6)
    led.record("final_answer", "gpt_5_4", 1500, 200, 2.0)
    g = led.grand_total()
    print("grand:", g)
    led.export("/tmp/test_ledger.json", include_raw=True)
    with open("/tmp/test_ledger.json") as f:
        doc = json.load(f)
    assert doc["totals"]["calls"] == 3
    assert doc["by_phase"]["distilled_text_gen"]["calls"] == 2
    assert doc["alias_status"] == "PROVISIONAL"
    print("[PASS] token_ledger self-test")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
