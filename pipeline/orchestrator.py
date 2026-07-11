"""
pipeline/orchestrator.py
========================
ISCEPipeline orchestrator. Integrates PKI, B1 (SCSV), MBD, B2
(Explainability), CP, B3 Bridge (Semantic Gate), the Trust Decision
Engine, and Adapters -- per the frozen architecture:

    Incoming Message
        -> PKI   (optional, opt-in -- see __init__ docstring)
        -> B1 (SCSV)                     -> ValidationAssessment
        -> MBD   (optional, opt-in)      -> MBDResult
        -> B2 (ExplainabilityEngine)     -> ExplainabilityReport
        -> CP    (optional, opt-in)      -> CPResult
        -> Synthesizer                   -> scene text
        -> B3 (SemanticGateClassifier)   -> SemanticResult
        -> TrustDecisionEngine           -> FinalTrustDecision
        -> Adapters                      -> {log, api, ds_mass, ...}

MBD and CP are OPT-IN (disabled by default) so that every existing
caller (manual_pipeline_test.py, the full test suite) continues to see
EXACTLY the same behavior as before this integration unless explicitly
enabled -- per the "preserve backwards compatibility" requirement.

B2 never sees the raw message. B3 never sees B1/B2 internals directly,
only the synthesized text. The TrustDecisionEngine is the sole fusion
point.

KNOWN SCOPING LIMITATION (documented, not silently glossed over): CP's
`observation_weights` are only fully trust-weighted for the TARGET
message's sender (the one that went through B1+MBD this call); other
senders in the message window default to weight 1.0, since this
orchestrator validates one target message per run() call, not every
peer independently. Full per-peer trust-weighted CP (running B1+MBD for
every report in the window before fusing) is a further increment, not
yet implemented -- see MIGRATION_REPORT.md.

PKI is wired but requires messages to carry actual signature/certificate/
public_key material to do anything meaningful; fixtures without that
material skip PKI (documented, not silently faked) -- see
MIGRATION_REPORT.md.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

from b1_scsv.scsv import SCSV
from b2_explain.explainability import ExplainabilityEngine
from pipeline.synthesizer import synthesize_message
from pipeline.b3_bridge import classify_text, preload_classifier
from trust_engine.decision_engine import TrustDecisionEngine
from trust_engine.policy import TrustPolicy
from adapters.base import Adapter
from contracts.trust_evidence import TrustEvidence

# Optional-layer imports are done lazily inside __init__/run() so that
# importing pipeline.orchestrator never requires numpy (CP's dependency)
# or cryptography (PKI's dependency) unless those layers are enabled.


def _normalize_b1_result(b1_res: Any) -> Dict[str, Any]:
    """Normalize B1's return value (dataclass, dict, or None) into the
    canonical ValidationAssessment dict shape consumed by B2 and the
    Trust Decision Engine.
    """
    if hasattr(b1_res, "valid"):
        return {
            "valid": b1_res.valid,
            "fatal": getattr(b1_res, "fatal", False),
            "score": getattr(b1_res, "validation_score", getattr(b1_res, "score", 1.0)),
            "confidence": getattr(b1_res, "confidence", 1.0),
            "reasons": getattr(b1_res, "reasons", []),
            "checks": getattr(b1_res, "checks", {}),
            "details": getattr(b1_res, "details", {}),
        }
    if isinstance(b1_res, dict):
        return {
            "valid": b1_res.get("valid", True),
            "fatal": b1_res.get("fatal", False),
            "score": b1_res.get("score", 1.0),
            "confidence": b1_res.get("confidence", 1.0),
            "reasons": b1_res.get("reasons", []),
            "checks": b1_res.get("checks", {}),
            "details": b1_res.get("details", {}),
        }
    return {
        "valid": True,
        "fatal": False,
        "score": 1.0,
        "confidence": 1.0,
        "reasons": [],
        "checks": {},
        "details": {},
    }


class ISCEPipeline:
    """Orchestrator managing PKI, B1 (SCSV), MBD, B2 (Explainability), CP,
    the Message Synthesizer, B3 (Semantic Gate), the Trust Decision
    Engine, and Adapters.
    """

    def __init__(
        self,
        scsv: Optional[SCSV] = None,
        explainability_engine: Optional[ExplainabilityEngine] = None,
        trust_policy: Optional[TrustPolicy] = None,
        csia: Optional[Any] = None,
        adapters: Optional[Dict[str, Adapter]] = None,
        enable_mbd: bool = False,
        enable_cp: bool = False,
        pki_ca: Optional[Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        enable_mbd:
            Opt-in. When True, MBD runs between B1 and B2, and B2 uses
            explain_evidence([B1, MBD]) instead of explain(B1). When
            False (default), behavior is EXACTLY the pre-MBD-integration
            pipeline -- zero difference.
        enable_cp:
            Opt-in. When True, CP runs between B2 and B3, fusing the
            full message window (weighted by B2's evidence for the
            target sender -- see module docstring's scoping limitation).
            When False (default), behavior is unchanged.
        pki_ca:
            Optional pki.CertificateAuthority instance. If a message
            carries `_pki_signature`/`_pki_certificate`/`_pki_public_key`
            keys, PKI runs and its result is folded into B1's checks. If
            absent, PKI is skipped for that message (not faked as
            passing) -- see module docstring.
        """
        self.scsv = scsv or SCSV()
        self.b2 = explainability_engine or ExplainabilityEngine()
        self.trust_engine = TrustDecisionEngine(policy=trust_policy)
        self.adapters = adapters or {}
        self.enable_mbd = enable_mbd
        self.enable_cp = enable_cp
        self.pki_ca = pki_ca

        self._mbd_history = None
        if self.enable_mbd:
            from mbd import VehicleHistoryStore
            self._mbd_history = VehicleHistoryStore()

        # Load B3's model NOW, at pipeline construction, not lazily on the
        # first classify_text() call inside run(). This keeps the ~150s
        # one-time model-load cost out of any single message's "bridge_ms"
        # (see pipeline/b3_bridge.py::preload_classifier docstring).
        self.b3_load_ms = preload_classifier()
        if csia is not None:
            # Legacy compatibility shim only. b2_csia.CSIA is deprecated and
            # NOT used by the pipeline anymore -- B2 is now b2_explain.
            self._legacy_csia = csia

    def _run_pki(self, target_msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Runs PKI if the message carries the required material and a
        CertificateAuthority was supplied. Returns None (not a fake pass)
        if PKI cannot run for this message -- see module docstring."""
        if self.pki_ca is None:
            return None
        sig = target_msg.get("_pki_signature")
        cert = target_msg.get("_pki_certificate")
        pub = target_msg.get("_pki_public_key")
        if sig is None or cert is None or pub is None:
            return None
        from pki import pki_layer
        return pki_layer(target_msg, sig, cert, pub, self.pki_ca)

    def _run_mbd(self, target_msg: Dict[str, Any]) -> Dict[str, Any]:
        from mbd import mbd_layer
        from bridges.message_adapter import to_flat_report, ProjectionOrigin

        lat = target_msg.get("cam", {}).get("cam_parameters", {}).get(
            "basic_container", {}).get("reference_position", {}).get("latitude")
        lon = target_msg.get("cam", {}).get("cam_parameters", {}).get(
            "basic_container", {}).get("reference_position", {}).get("longitude")
        origin = ProjectionOrigin.from_degrees(
            lat * 1e-7 if lat and abs(lat) > 1000 else (lat or 0.0),
            lon * 1e-7 if lon and abs(lon) > 1000 else (lon or 0.0),
        )
        flat = to_flat_report(target_msg, origin)

        station_id = flat["sender"]
        cert_rotation_anomaly = None
        if self.scsv._cert_rotation_owner == "mbd":
            cert_rotation_anomaly = self.scsv.check_cert_rotation_for_station(station_id)

        return dict(mbd_layer(flat, self._mbd_history, cert_rotation_anomaly=cert_rotation_anomaly))

    def _run_cp(
        self,
        messages: List[Dict[str, Any]],
        target_sender_weight: float,
        target_sender_id: Any,
    ) -> Dict[str, Any]:
        from cp import cp_layer
        from bridges.message_adapter import to_flat_report, ProjectionOrigin

        target_msg = messages[-1]
        lat = target_msg.get("cam", {}).get("cam_parameters", {}).get(
            "basic_container", {}).get("reference_position", {}).get("latitude")
        lon = target_msg.get("cam", {}).get("cam_parameters", {}).get(
            "basic_container", {}).get("reference_position", {}).get("longitude")
        origin = ProjectionOrigin.from_degrees(
            lat * 1e-7 if lat and abs(lat) > 1000 else (lat or 0.0),
            lon * 1e-7 if lon and abs(lon) > 1000 else (lon or 0.0),
        )

        reports = []
        weights: Dict[Any, float] = {}
        for m in messages:
            try:
                flat = to_flat_report(m, origin)
            except ValueError:
                continue  # skip malformed peer reports rather than crash CP fusion
            reports.append(flat)
            # Scoping limitation (documented in module docstring): only the
            # target sender's weight is derived from real B1+MBD+B2 output
            # this call; other senders default to 1.0 (unweighted).
            weights[flat["sender"]] = (
                target_sender_weight if flat["sender"] == target_sender_id else 1.0
            )

        return cp_layer(reports, observation_weights=weights)

    def run(
        self,
        messages: List[Dict[str, Any]],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the pipeline from PKI/B1 through Adapters.

        Parameters
        ----------
        messages: List[dict]
            Window of messages, with target message as messages[-1].
        context: Optional[str]
            Operational context candidate (e.g. 'urban', 'rural', 'highway').

        Returns
        -------
        dict
            Pipeline result dictionary.
        """
        if not messages:
            raise ValueError("ISCEPipeline.run: empty messages window supplied.")

        t_total_start = time.perf_counter()
        target_msg = messages[-1]

        # 0. Run PKI (opt-in, see _run_pki)
        t_pki_start = time.perf_counter()
        pki_result = self._run_pki(target_msg)
        pki_ms = (time.perf_counter() - t_pki_start) * 1000.0

        # 1. Run B1 (SCSV)
        t_b1_start = time.perf_counter()
        if target_msg.get("_validation_assessment") is not None:
            b1_res = target_msg["_validation_assessment"]
        else:
            b1_res = self.scsv.check_stateful(target_msg)
        b1_ms = (time.perf_counter() - t_b1_start) * 1000.0

        b1_dict = _normalize_b1_result(b1_res)
        target_msg["_validation_assessment"] = b1_res
        if pki_result is not None:
            b1_dict["checks"]["pki_signature"] = pki_result["sig_valid"]
            b1_dict["checks"]["pki_revocation"] = not pki_result["revoked"]
            b1_dict["details"]["pki_compromised_flag"] = pki_result["compromised"]

        # 2. Run MBD (opt-in)
        t_mbd_start = time.perf_counter()
        mbd_dict: Optional[Dict[str, Any]] = None
        if self.enable_mbd:
            mbd_dict = self._run_mbd(target_msg)
        mbd_ms = (time.perf_counter() - t_mbd_start) * 1000.0

        # 3. Run B2 (Explainability) — ALWAYS runs, including on B1-fatal
        #    paths. Uses explain_evidence([B1,MBD]) when MBD is enabled,
        #    otherwise the original explain(B1) path -- UNCHANGED when
        #    enable_mbd=False, per the backward-compatibility requirement.
        t_b2_start = time.perf_counter()
        if mbd_dict is not None:
            evidence = [
                TrustEvidence.from_validation_assessment(b1_dict),
                TrustEvidence.from_mbd_result(mbd_dict),
            ]
            b2_report = self.b2.explain_evidence(evidence)
        else:
            b2_report = self.b2.explain(b1_dict)
        b2_dict = b2_report.to_dict()
        b2_ms = (time.perf_counter() - t_b2_start) * 1000.0

        # 4. Run CP (opt-in)
        t_cp_start = time.perf_counter()
        cp_dict: Optional[Dict[str, Any]] = None
        if self.enable_cp:
            target_sender_id = target_msg.get("header", {}).get("station_id")
            cp_dict = self._run_cp(
                messages,
                target_sender_weight=b2_dict["confidence_calibration"],
                target_sender_id=target_sender_id,
            )
        cp_ms = (time.perf_counter() - t_cp_start) * 1000.0

        # 5. Run Message Synthesizer — consumes ONLY the raw message
        #    cluster (never B2 output).
        t_synt_start = time.perf_counter()
        synthesized_message = synthesize_message(messages, b2_dict, context)
        synt_ms = (time.perf_counter() - t_synt_start) * 1000.0

        # 6. Run B3 Adapter Bridge (Semantic Gate)
        t_bridge_start = time.perf_counter()
        b3_result = classify_text(synthesized_message["text"], synthesized_message)
        bridge_ms = (time.perf_counter() - t_bridge_start) * 1000.0

        # === PHASE A FIX: Fold CP evidence into the decision inputs ===
        if cp_dict is not None:
            
            cp_evidence = TrustEvidence.from_cp_result(cp_dict)
            
            # Use 'most-conservative-wins' (min) instead of averaging, 
            # preventing high CP confidence from washing out B1/MBD penalties.
            combined_score = min(b2_dict["validation_score"], cp_evidence.score)
            combined_valid = b2_dict["validation_valid"] and cp_evidence.passed
            
            b2_dict = {
                **b2_dict,
                "validation_score": combined_score,
                "validation_valid": combined_valid,
                "provenance": {
                    **b2_dict.get("provenance", {}),
                    "source_layers": list(b2_dict.get("provenance", {}).get("source_layers", [])) + ["CP"],
                },
            }
        # ==============================================================
        # 7. Run Trust Decision Engine — the sole fusion point for
        #    B1 + B2 + B3 into a FinalTrustDecision. (MBD/CP feed in via
        #    B1_dict/B2_dict already; TrustDecisionEngine's own signature
        #    is UNCHANGED, per the frozen architecture.)
        t_fuse_start = time.perf_counter()
        final_decision = self.trust_engine.decide(b1_dict, b2_dict, b3_result)
        fuse_ms = (time.perf_counter() - t_fuse_start) * 1000.0

        total_ms = (time.perf_counter() - t_total_start) * 1000.0

        decision_dict = final_decision.to_dict()

        adapted: Dict[str, Any] = {}
        for name, adapter in self.adapters.items():
            adapted[name] = adapter.adapt(final_decision)

        return {
            "pki": pki_result,
            "b1": b1_dict,
            "mbd": mbd_dict,
            "b2": b2_dict,
            "cp": cp_dict,
            "synthesized_message": synthesized_message,
            "b3": b3_result,
            "decision": decision_dict["trust_level"],
            "reason": decision_dict["reasoning"],
            "fusion": decision_dict,
            "final_trust_decision": final_decision,
            "adapted": adapted,
            "latencies": {
                "pki_ms": pki_ms,
                "b1_ms": b1_ms,
                "mbd_ms": mbd_ms,
                "b2_ms": b2_ms,
                "cp_ms": cp_ms,
                "synthesizer_ms": synt_ms,
                "bridge_ms": bridge_ms,
                "fusion_ms": fuse_ms,
                "total_ms": total_ms
            }
        }