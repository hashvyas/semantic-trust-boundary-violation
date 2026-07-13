"""
semantic_evaluation/semantic_attack_evaluation.py
====================================================
Three-way pipeline evaluation: B1-only vs B1+B2 vs B1+B2+B3 (full).

Runs every scenario from semantic_attack_dataset.py through each
pipeline configuration and records per-scenario results for downstream
metric computation.

All ablations use monkey-patching on the existing ISCEPipeline
(same pattern as evaluation/runner.py) — no production code is modified.
"""

from __future__ import annotations

import pathlib
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from b1_scsv.scsv import SCSV
from pipeline.orchestrator import ISCEPipeline
import pipeline.orchestrator as _orch_module
from pipeline.b3_bridge import preload_classifier

from semantic_evaluation.semantic_attack_dataset import (
    ALL_SCENARIOS,
    SemanticAttackScenario,
)
from semantic_evaluation.semantic_attack_generator import generate_corpus

# ----------------------------------------------------------------
# Pipeline configurations for the three-way comparison
# ----------------------------------------------------------------
SEMANTIC_CONFIGURATIONS = ["b1_only", "b1_b2", "full"]


def _build_pipeline(configuration: str) -> Tuple[ISCEPipeline, Callable[[], None]]:
    """Build an ISCEPipeline with the requested ablation.

    Returns (pipeline, restore_fn).  restore_fn undoes any
    module-level monkey-patching.
    """
    # All configurations disable MBD and CP to isolate the
    # structural-only vs semantic comparison cleanly.
    pipe = ISCEPipeline(
        scsv=SCSV(),
        enable_mbd=False,
        enable_cp=False,
        pki_ca=None,
    )
    restores: List[Callable[[], None]] = []

    if configuration == "b1_only":
        # Neuter B2 to passthrough AND force B3 unavailable.
        from b2_explain.models import ExplainabilityReport

        orig_explain = pipe.b2.explain
        def fake_explain(va: Dict[str, Any]) -> ExplainabilityReport:
            return ExplainabilityReport(
                explanation_text="[B2 ABLATED]",
                evidence=[],
                confidence_calibration=va.get("confidence", 1.0),
                provenance={},
                validation_valid=va.get("valid", True),
                validation_score=va.get("score", 1.0),
            )
        pipe.b2.explain = fake_explain
        restores.append(lambda: setattr(pipe.b2, "explain", orig_explain))

        # Force B3 unavailable
        original_ct = _orch_module.classify_text
        _orch_module.classify_text = lambda text, metadata=None: {
            "available": False, "label": None, "confidence": None,
            "risk_level": "unavailable", "status": "ABLATED (b1_only)",
            "p_malicious": None,
        }
        restores.append(lambda: setattr(_orch_module, "classify_text", original_ct))

    elif configuration == "b1_b2":
        # B2 runs normally; force B3 unavailable.
        original_ct = _orch_module.classify_text
        _orch_module.classify_text = lambda text, metadata=None: {
            "available": False, "label": None, "confidence": None,
            "risk_level": "unavailable", "status": "ABLATED (b1_b2)",
            "p_malicious": None,
        }
        restores.append(lambda: setattr(_orch_module, "classify_text", original_ct))

    elif configuration == "full":
        pass  # No ablation — B1 + B2 + B3 all active.

    def restore_all() -> None:
        for r in reversed(restores):
            r()

    return pipe, restore_all


def evaluate_scenario(
    pipe: ISCEPipeline,
    msg: Dict[str, Any],
    configuration: str,
) -> Dict[str, Any]:
    """Run a single message through the pipeline and record results."""
    row: Dict[str, Any] = {
        "attack_id": msg.get("attack_id"),
        "attack_category": msg.get("attack_category"),
        "attack_subcategory": msg.get("attack_subcategory"),
        "difficulty": msg.get("difficulty"),
        "expected_label": msg.get("expected_label"),
        "truth_attacker": msg.get("is_attacker", False),
        "configuration": configuration,
    }
    try:
        t0 = time.perf_counter()
        result = pipe.run([msg])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        row.update({
            "decision": result["decision"],
            "trust_score": result["fusion"]["trust_score"],
            "b3_available": result["b3"]["available"],
            "b3_label": result["b3"].get("label"),
            "b3_confidence": result["b3"].get("confidence"),
            "b3_risk_level": result["b3"].get("risk_level"),
            "b3_p_malicious": result["b3"].get("p_malicious"),
            "semantic_risk": result["fusion"].get("semantic_risk"),
            "cryptographic_risk": result["fusion"].get("cryptographic_risk"),
            "b1_score": result["b1"].get("score", 1.0),
            "b1_valid": result["b1"].get("valid", True),
            "b1_fatal": result["b1"].get("fatal", False),
            "synthesized_text": result.get("synthesized_message", {}).get("text", ""),
            "latencies": result.get("latencies", {}),
            "total_ms": elapsed_ms,
            "reasoning": result["fusion"].get("reasoning", ""),
            "error": None,
        })
    except Exception as e:
        row.update({
            "decision": "ERROR",
            "trust_score": None,
            "b3_available": None,
            "b3_label": None,
            "b3_confidence": None,
            "b3_risk_level": None,
            "b3_p_malicious": None,
            "semantic_risk": None,
            "cryptographic_risk": None,
            "b1_score": None,
            "b1_valid": None,
            "b1_fatal": None,
            "synthesized_text": "",
            "latencies": {},
            "total_ms": None,
            "reasoning": "",
            "error": f"{type(e).__name__}: {e}",
        })
    return row


def run_evaluation(
    configurations: Optional[List[str]] = None,
    scenarios: Optional[List[SemanticAttackScenario]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run the full three-way evaluation.

    Parameters
    ----------
    configurations : list of str, optional
        Pipeline configurations to evaluate (default: all three).
    scenarios : list of SemanticAttackScenario, optional
        Subset of scenarios (default: entire corpus).
    seed : int
        RNG seed for message generation.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    list of dict
        One row per (scenario, configuration) pair.
    """
    if configurations is None:
        configurations = list(SEMANTIC_CONFIGURATIONS)
    if scenarios is None:
        scenarios = list(ALL_SCENARIOS)

    # Pre-generate all messages (deterministic, same for all configs)
    messages = generate_corpus(scenarios, seed=seed)

    all_rows: List[Dict[str, Any]] = []

    for config in configurations:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Configuration: {config}")
            print(f"  Scenarios:     {len(messages)}")
            print(f"{'='*60}")

        pipe, restore = _build_pipeline(config)
        try:
            for i, (scenario, msg) in enumerate(zip(scenarios, messages)):
                row = evaluate_scenario(pipe, msg, config)
                all_rows.append(row)
                if verbose and (i + 1) % 20 == 0:
                    print(f"  [{config}] {i+1}/{len(messages)} done...")
        finally:
            restore()

        if verbose:
            n_rows = [r for r in all_rows if r["configuration"] == config]
            n_reject = sum(1 for r in n_rows if r["decision"] == "REJECT")
            n_caution = sum(1 for r in n_rows if r["decision"] == "CAUTION")
            n_accept = sum(1 for r in n_rows if r["decision"] == "ACCEPT")
            n_error = sum(1 for r in n_rows if r["decision"] == "ERROR")
            print(f"  [{config}] Complete: REJECT={n_reject} CAUTION={n_caution} "
                  f"ACCEPT={n_accept} ERROR={n_error}")

    return all_rows


if __name__ == "__main__":
    import json
    # Quick standalone test
    preload_classifier()
    rows = run_evaluation(verbose=True)
    print(f"\nTotal rows: {len(rows)}")
    print(json.dumps(rows[:2], indent=2, default=str))
