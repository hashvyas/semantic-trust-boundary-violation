"""
pipeline/b3_bridge.py
======================
Stable adapter bridge module for B3 semantic classification.
Provides integration contract and default stub implementation.

Owns the SemanticResult contract, INCLUDING risk_level. B3 is the
semantic-reasoning layer, so B3 -- not the Trust Decision Engine -- owns
the taxonomy of what counts as a "malicious" label and what confidence
bands map to what risk level. The Trust Decision Engine consumes
risk_level as an opaque field of B3's public output; it does not know
or care what label strings B3's underlying model uses.
"""

from __future__ import annotations
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional
import yaml

_DEFAULT_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "isce_config.yaml"


@dataclass(frozen=True)
class SemanticResult:
    """B3's typed public output. This is the ONLY contract other layers
    (specifically trust_engine.decision_engine.TrustDecisionEngine) may
    depend on. No caller may reach past this into the predictor, model,
    or tokenizer.

    available:
        False when the classifier could not run (missing deps, missing
        checkpoint, inference error). Callers must treat this as
        "no semantic signal", not as "benign".
    label:
        Raw (but already-normalized) classifier label, or None when
        unavailable. Kept for logging/forensics; NOT meant to be
        pattern-matched by callers outside this module -- use risk_level.
    confidence:
        Classifier softmax confidence in [0, 1], or None when unavailable.
    risk_level:
        Normalized semantic risk band: "none" | "low" | "medium" | "high"
        | "unavailable". This is B3's actual public contract -- computed
        here, from B3's own configured thresholds, not by the Trust
        Decision Engine.
    status:
        Human-readable status/diagnostic string.
    """

    available: bool
    label: Optional[str]
    confidence: Optional[float]
    risk_level: str
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "label": self.label,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "status": self.status,
        }

    @staticmethod
    def unavailable(status: str) -> "SemanticResult":
        return SemanticResult(
            available=False, label=None, confidence=None,
            risk_level="unavailable", status=status,
        )


@dataclass(frozen=True)
class B3RiskPolicy:
    """B3's own risk-banding configuration. Distinct from (and unrelated
    to) trust_engine.policy.TrustPolicy's cryptographic score bands --
    this only governs how B3 maps its own label+confidence to risk_level.
    """

    malicious_labels: FrozenSet[str] = frozenset({"MALICIOUS", "MALICIOUS_SEMANTIC_MANIPULATION"})
    high_confidence: float = 0.85
    medium_confidence: float = 0.60

    def classify(self, label: Optional[str], confidence: Optional[float]) -> str:
        if label is None or confidence is None:
            return "unavailable"
        if label not in self.malicious_labels:
            return "none"
        if confidence >= self.high_confidence:
            return "high"
        if confidence >= self.medium_confidence:
            return "medium"
        return "low"

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "B3RiskPolicy":
        thresholds = config.get("risk_thresholds", {}) or {}
        labels = config.get("malicious_labels")
        return B3RiskPolicy(
            malicious_labels=frozenset(labels) if labels else B3RiskPolicy().malicious_labels,
            high_confidence=thresholds.get("high", 0.85),
            medium_confidence=thresholds.get("medium", 0.60),
        )


def resolve_model_path(model_path: str) -> str:
    """Resolve model path against absolute path and the workspace root."""
    if os.path.exists(model_path):
        return os.path.abspath(model_path)
    # Try relative to workspace root (parent of pipeline dir)
    workspace_root = pathlib.Path(__file__).resolve().parent.parent
    candidate = workspace_root / model_path
    if candidate.exists():
        return os.path.abspath(candidate)
    return os.path.abspath(model_path)

def _load_b3_config(config_path: Optional[str | os.PathLike] = None) -> Dict[str, Any]:
    """Load configuration from isce_config.yaml."""
    path = pathlib.Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data.get("b3_semantic_gate", {})
    except Exception:
        return {}

class StubSemanticClassifier:
    """Default stub classifier returning 'unavailable' state.
    Allows testing/running the pipeline without an actual B3 model dependency.
    """
    def classify(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return SemanticResult.unavailable("B3 integration unavailable").to_dict()

class SemanticGateClassifier:
    """Real B3 classifier wrapper loading the trained DeBERTa model and performing inference."""

    # Multiple shapes warmed (0/1/2/3 cluster peers) because a single fixed
    # warmup shape left later, differently-shaped real messages (e.g. msg 3
    # in benign fixtures) measurably slower than others -- reproduced across
    # 3 independent runs (268/302/252ms vs ~30-100ms for other messages).
    # Warming only one shape does not amortize CUDA/cuDNN kernel-selection
    # cost for shapes that differ from it.
    _WARMUP_TEMPLATES = [
        "V2X Scene Report: context=rural. Ego vehicle: station 0000 "
        "(type=passengerCar), position=(lat=0, lon=0), speed=0, "
        "heading=0 deg, yaw_rate=0, longitudinal_acceleration=0, "
        "timestamp=0.0. Local sensor observations: camera=UNKNOWN, "
        "radar=UNKNOWN, lidar=UNKNOWN. No peer reports received. "
        "No RSU messages received. No other vehicles in cooperative cluster.",
        "V2X Scene Report: context=rural. Ego vehicle: station 0000 "
        "(type=passengerCar), position=(lat=0, lon=0), speed=0, "
        "heading=0 deg, yaw_rate=0, longitudinal_acceleration=0, "
        "timestamp=0.0. Local sensor observations: camera=UNKNOWN, "
        "radar=UNKNOWN, lidar=UNKNOWN. No RSU messages received. "
        "Cluster peer 1, (station 0001, type=passengerCar), "
        "position=(lat=0, lon=0), distance=1.0 m from ego, speed=0, "
        "heading=0 deg, yaw_rate=0, timestamp=0.0.",
        "V2X Scene Report: context=rural. Ego vehicle: station 0000 "
        "(type=passengerCar), position=(lat=0, lon=0), speed=0, "
        "heading=0 deg, yaw_rate=0, longitudinal_acceleration=0, "
        "timestamp=0.0. Local sensor observations: camera=UNKNOWN, "
        "radar=UNKNOWN, lidar=UNKNOWN. No RSU messages received. "
        "Cluster peer 1, (station 0001, type=passengerCar), "
        "position=(lat=0, lon=0), distance=1.0 m from ego, speed=0, "
        "heading=0 deg, yaw_rate=0, timestamp=0.0. Cluster peer 2, "
        "(station 0002, type=heavyTruck), position=(lat=0, lon=0), "
        "distance=2.0 m from ego, speed=0, heading=0 deg, yaw_rate=0, "
        "timestamp=0.0.",
        "V2X Scene Report: context=rural. Ego vehicle: station 0000 "
        "(type=roadSideUnit), position=(lat=0, lon=0), speed=N/A, "
        "heading=N/A deg, yaw_rate=0, longitudinal_acceleration=0, "
        "timestamp=0.0. Local sensor observations: camera=UNKNOWN, "
        "radar=UNKNOWN, lidar=UNKNOWN. No RSU messages received. "
        "Cluster peer 1, (station 0001, type=passengerCar), "
        "position=(lat=0, lon=0), distance=1.0 m from ego, speed=0, "
        "heading=0 deg, yaw_rate=0, timestamp=0.0. Cluster peer 2, "
        "(station 0002, type=heavyTruck), position=(lat=0, lon=0), "
        "distance=2.0 m from ego, speed=0, heading=0 deg, yaw_rate=0, "
        "timestamp=0.0. Cluster peer 3, (station 0003, type=motorcycle), "
        "position=(lat=0, lon=0), distance=3.0 m from ego, speed=0, "
        "heading=0 deg, yaw_rate=0, timestamp=0.0.",
    ]
    _WARMUP_ROUNDS = 3  # each template warmed 3x -> 12 total calls, still
                        # within the one-time startup window excluded from
                        # per-message latency

    def __init__(self, config_path: Optional[str | os.PathLike] = None) -> None:
        import time as _time
        _load_start = _time.perf_counter()

        self.config = _load_b3_config(config_path)
        self.model_path = self.config.get("model_path", "b3/solution_stb/b3_semantic_gate/model/semantic_gate_v3")
        self.max_length = self.config.get("max_length", 256)
        self.device = self.config.get("device", None)
        self.risk_policy = B3RiskPolicy.from_config(self.config)
        self.predictor = None
        self.error_status = None
        self.load_time_ms: float = 0.0
        
        # Verify model directory exists
        resolved_path = resolve_model_path(self.model_path)
        if not os.path.exists(resolved_path):
            self.error_status = f"B3 model checkpoint not found at {resolved_path}"
            self.load_time_ms = (_time.perf_counter() - _load_start) * 1000.0
            return

        try:
            # Ensure workspace is in sys.path so b3 package is importable
            workspace_root = str(pathlib.Path(__file__).resolve().parent.parent)
            if workspace_root not in sys.path:
                sys.path.insert(0, workspace_root)
                
            from b3.solution_stb.b3_semantic_gate.inference import get_predictor
            self.predictor = get_predictor(self.model_path, max_length=self.max_length, device=self.device)

            # Warmup: absorb CUDA kernel-selection/GPU-passthrough overhead
            # here, at startup, not during the first real messages. Wrapped
            # in try/except -- if warmup itself fails for any reason, the
            # classifier is still usable (just cold on message 1), so a
            # warmup failure must never take down B3 availability.
            try:
                for _ in range(self._WARMUP_ROUNDS):
                    for template in self._WARMUP_TEMPLATES:
                        self.predictor.predict([template])
            except Exception:
                pass  # warmup is a latency optimization, not a correctness dependency

        except Exception as e:
            self.error_status = f"Failed to initialize B3 predictor: {str(e)}"
        finally:
            self.load_time_ms = (_time.perf_counter() - _load_start) * 1000.0

    def classify(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.error_status or self.predictor is None:
            return SemanticResult.unavailable(
                self.error_status or "B3 predictor uninitialized"
            ).to_dict()
        try:
            results = self.predictor.predict([message])
            if not results:
                return SemanticResult.unavailable("Inference returned empty results").to_dict()
            res = results[0]
            # Standardize label mapping (map MALICIOUS_SEMANTIC_MANIPULATION to MALICIOUS)
            label_name = "MALICIOUS" if res.label == "MALICIOUS_SEMANTIC_MANIPULATION" else res.label
            risk_level = self.risk_policy.classify(label_name, res.confidence)
            return SemanticResult(
                available=True,
                label=label_name,
                confidence=res.confidence,
                risk_level=risk_level,
                status="ok",
            ).to_dict()
        except Exception as e:
            return SemanticResult.unavailable(f"Inference execution error: {str(e)}").to_dict()

_CLASSIFIER_INSTANCE: Optional[SemanticGateClassifier] = None

def preload_classifier(config_path: Optional[str | os.PathLike] = None) -> float:
    """Eagerly construct (and thus load) the B3 classifier singleton, OUTSIDE
    of any per-message timing block.

    ROOT CAUSE THIS FIXES: previously the classifier was constructed lazily
    on the first call to classify_text(), which meant the ~150s one-time
    cost of loading the DeBERTa checkpoint + tokenizer got silently counted
    as "bridge_ms" (per-message B3 latency) for whichever message happened
    to be first through the pipeline -- making that message's latency look
    150,000x worse than every subsequent one, and making the reported
    "Average Total Latency" over a batch meaningless (it's actually
    ~(model_load_time + N*real_inference_time) / N, not a real per-message
    average).

    Call this once at pipeline/application startup (ISCEPipeline.__init__
    now does this automatically). Safe to call multiple times -- only the
    first call actually loads anything; subsequent calls are no-ops.

    Returns the load time in milliseconds (0.0 if already loaded).
    """
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is None:
        _CLASSIFIER_INSTANCE = SemanticGateClassifier(config_path)
        return _CLASSIFIER_INSTANCE.load_time_ms
    return 0.0

def classify_text(message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Interfacing function for B3 semantic classifier.

    Returns a dict matching SemanticResult.to_dict() -- keys:
    available, label, confidence, risk_level, status.

    NOTE: if preload_classifier() was not called first, this will still
    work correctly (lazy-load fallback preserved for backward
    compatibility / ad-hoc scripts), but the first call's timing will
    include model load time -- exactly the behavior preload_classifier()
    exists to avoid in a real pipeline run.
    """
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is None:
        _CLASSIFIER_INSTANCE = SemanticGateClassifier()
    return _CLASSIFIER_INSTANCE.classify(message, metadata)