from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from pipeline.b3_bridge import B3RiskPolicy
from pipeline.orchestrator import ISCEPipeline

def test_risk_policy_confidence_aware_benign():
    # Test default policy behavior (confidence_aware_benign = False)
    policy_default = B3RiskPolicy(confidence_aware_benign=False)
    assert policy_default.classify("BENIGN", 0.99) == "none"
    assert policy_default.classify("BENIGN", 0.54) == "none"
    
    # Test confidence_aware_benign = True
    policy_aware = B3RiskPolicy(
        high_confidence=0.85,
        medium_confidence=0.60,
        confidence_aware_benign=True
    )
    # High confidence benign -> none
    assert policy_aware.classify("BENIGN", 0.90) == "none"
    # Medium confidence benign -> low
    assert policy_aware.classify("BENIGN", 0.70) == "low"
    # Low confidence benign -> medium
    assert policy_aware.classify("BENIGN", 0.54) == "medium"
    
    # Malicious labels should still follow regular rules
    assert policy_aware.classify("MALICIOUS", 0.90) == "high"
    assert policy_aware.classify("MALICIOUS", 0.70) == "medium"
    assert policy_aware.classify("MALICIOUS", 0.54) == "low"


@patch("pipeline.orchestrator.classify_text")
@patch("pipeline.orchestrator.synthesize_message")
def test_pipeline_ensembling_logic(mock_synthesize, mock_classify):
    # Setup mock outputs for three templates
    # TemplateStyle: DEFAULT, NARRATIVE, STRUCTURED
    mock_synthesize.side_effect = [
        {"text": "default text", "template_style": "default"},
        {"text": "narrative text", "template_style": "narrative"},
        {"text": "structured text", "template_style": "structured"},
    ]
    
    # Mock classifier outputs with different p_malicious
    # 1. DEFAULT: BENIGN, conf 0.99 -> p_malicious = 0.01
    # 2. NARRATIVE: BENIGN, conf 0.54 -> p_malicious = 0.46
    # 3. STRUCTURED: MALICIOUS, conf 0.95 -> p_malicious = 0.95
    # Avg p_malicious = (0.01 + 0.46 + 0.95) / 3 = 1.42 / 3 = 0.473
    # Since 0.473 < 0.5, label is BENIGN, conf is 1 - 0.473 = 0.527
    mock_classify.side_effect = [
        {"available": True, "label": "BENIGN", "confidence": 0.99, "p_malicious": 0.01},
        {"available": True, "label": "BENIGN", "confidence": 0.54, "p_malicious": 0.46},
        {"available": True, "label": "MALICIOUS", "confidence": 0.95, "p_malicious": 0.95},
    ]
    
    pipeline = ISCEPipeline()
    pipeline.enable_b3_ensembling = True
    
    # Configure B3RiskPolicy mock to check classification call
    mock_policy = MagicMock()
    mock_policy.classify.return_value = "low"
    
    with patch("pipeline.b3_bridge._CLASSIFIER_INSTANCE") as mock_classifier_instance:
        mock_classifier_instance.risk_policy = mock_policy
        
        # Run orchestrator
        msg = {
            "station_id": 1001,
            "latitude": 0.0,
            "longitude": 0.0,
            "speed": 0.0,
            "heading": 0.0,
            "timestamp": 0.0,
            "_validation_assessment": {
                "valid": True,
                "fatal": False,
                "score": 1.0,
                "confidence": 1.0,
                "reasons": [],
                "checks": {},
                "details": {}
            }
        }
        res = pipeline.run(
            messages=[msg], 
            context="urban"
        )
        
        # Verify average calculation and final classification
        b3_res = res["b3"]
        assert b3_res["available"]
        assert b3_res["label"] == "BENIGN"
        assert pytest.approx(b3_res["confidence"], abs=1e-4) == 0.5266
        assert b3_res["risk_level"] == "low"
        assert pytest.approx(b3_res["p_malicious"], abs=1e-4) == 0.4733
        
        # Check that risk policy classify was invoked with aggregated results
        mock_policy.classify.assert_called_once_with("BENIGN", pytest.approx(0.5266, abs=1e-4))
