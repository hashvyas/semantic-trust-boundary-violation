"""
b2_explain/models.py
=====================
Typed data models for the B2 Explainability layer.

B2 explains B1's ValidationAssessment. It never performs semantic
interpretation of the raw V2X payload (that is B3's exclusive
responsibility) and it never overrides B1's verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from b2_explain.evidence import EvidenceDirection, EvidenceItem

__all__ = ["EvidenceDirection", "EvidenceItem", "ExplainabilityReport"]


@dataclass(frozen=True)
class ExplainabilityReport:
    """B2's output. A pure function of B1's ValidationAssessment.

    Never mutates or re-scores ``valid``/``fatal``/``score`` from B1 —
    those fields are carried through unchanged in ``validation_score``
    and ``validation_valid`` purely for traceability/display.
    """

    explanation_text: str
    evidence: List[EvidenceItem] = field(default_factory=list)
    confidence_calibration: float = 1.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    # Read-only passthrough of the B1 verdict this report explains.
    validation_valid: bool = True
    validation_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explanation_text": self.explanation_text,
            "evidence": [
                {
                    "factor": e.factor,
                    "description": e.description,
                    "weight": e.weight,
                    "direction": e.direction.value,
                    "source_field": e.source_field,
                }
                for e in self.evidence
            ],
            "confidence_calibration": self.confidence_calibration,
            "provenance": self.provenance,
            "validation_valid": self.validation_valid,
            "validation_score": self.validation_score,
        }
