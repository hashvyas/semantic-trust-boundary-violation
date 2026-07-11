"""
validation/run_b3_b6_synthesizer_robustness.py
=================================================
Roadmap B3 (adversarial robustness of the synthesis interface) + B6
(template sensitivity analysis) -- combined into one script since both
exercise the same lever: pipeline/synthesizer.py's three interchangeable
TemplateStyle renderers (DEFAULT / NARRATIVE / STRUCTURED), which
already exist in this codebase for exactly this purpose (see that
module's own docstring).

What CAN be validated without GPU/torch (runs fully in this
environment):
  1. A2-leak-freedom holds across ALL THREE templates, not just the
     default one (a leak fixed for one template but not the others
     would be a real, easy-to-miss bug).
  2. Determinism: identical structured input -> identical text output,
     per template, every time.
  3. Structural diversity: the three templates actually produce
     meaningfully different text for the same input (otherwise "3
     templates" is cosmetic, not a real sensitivity test).

What requires GPU/torch (real B3 model) and is reported honestly as
blocked here, with the exact call to make once available:
  4. B6: does B3's label/confidence stay stable across all 3 templates
     for the same underlying facts? (Answers "did you just get lucky
     with one template's wording.")
  5. B3: does an adversarially-constructed structured input produce
     synthesized text that sits in a blind spot -- i.e. B1/B2 alone
     would not catch the attack, but no template's synthesized text
     triggers B3 either?

Run with: python3 validation/run_b3_b6_synthesizer_robustness.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.synthesizer import synthesize_message, TemplateStyle
from pipeline.b3_bridge import classify_text, preload_classifier

_B3_LOAD_MS = preload_classifier()
_PROBE = classify_text("capability probe")
B3_AVAILABLE = bool(_PROBE.get("available"))

_FAILURES = []


def check(name, cond, evidence=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {evidence}" if evidence else ""))
    if not cond:
        _FAILURES.append(name)


print("=" * 100)
print("ROADMAP B3 (adversarial synthesis robustness) + B6 (template sensitivity)")
print("=" * 100)
print(f"Real B3 model available in this run: {B3_AVAILABLE}")
print("=" * 100)

BANNED_SUBSTRINGS = [
    "trust", "belief", "disbelief", "uncertainty", "cluster_score", "entropy",
    "replay_probability", "identity_consistency", "confidence_calibration",
]

# Fake b2_result with obviously-identifiable B2-derived values -- if any of
# these numbers or the banned substrings above leak into synthesized text
# for ANY template, that's an A2/B3 regression.
_LEAKY_B2_RESULT = {
    "validation_valid": True, "validation_score": 0.1234567,
    "confidence_calibration": 0.7654321,
    "explanation_text": "LEAK_CANARY_EXPLANATION_TEXT_314159",
    "evidence": [], "provenance": {}, "belief": 0.271828, "disbelief": 0.161803,
    "uncertainty": 0.099999, "cluster_score": 0.888888, "entropy": 0.777777,
}

cluster_dir = ROOT / "scenarios" / "collusion"
cluster = [json.loads(f.read_text()) for f in sorted(cluster_dir.glob("*.json"))[:5]]

print("\n--- 1. A2-leak-freedom across all three templates ---")
texts_by_template: Dict[str, str] = {}
for style in TemplateStyle:
    result = synthesize_message(cluster, _LEAKY_B2_RESULT, context="urban", template=style)
    text = result["text"]
    texts_by_template[style.name] = text
    for banned in BANNED_SUBSTRINGS:
        check(f"[{style.name}] does not contain banned term '{banned}'", banned not in text.lower())
    for canary in ("0.1234567", "0.7654321", "LEAK_CANARY", "0.271828", "0.161803", "0.888888", "0.777777"):
        check(f"[{style.name}] does not contain B2-derived canary value '{canary}'", canary not in text)

print("\n--- 2. Determinism per template ---")
for style in TemplateStyle:
    r1 = synthesize_message(cluster, _LEAKY_B2_RESULT, context="urban", template=style)
    r2 = synthesize_message(cluster, _LEAKY_B2_RESULT, context="urban", template=style)
    check(f"[{style.name}] identical input -> identical output text", r1["text"] == r2["text"])

print("\n--- 3. Structural diversity across templates (same facts, different wording) ---")
unique_texts = set(texts_by_template.values())
check("All three templates produce distinct text for the same input (not cosmetic-only)",
      len(unique_texts) == len(texts_by_template),
      f"{len(unique_texts)} distinct outputs out of {len(texts_by_template)} templates")
for name, text in texts_by_template.items():
    print(f"  [{name}] ({len(text)} chars): {text[:160]}{'...' if len(text) > 160 else ''}")

print("\n--- 4/5. B6 template-stability + B3 adversarial blind-spot check (needs real B3) ---")
if not B3_AVAILABLE:
    print("BLOCKED in this environment (no torch/GPU). Exact calls to run once B3 is available:")
    print("""
    from pipeline.synthesizer import synthesize_message, TemplateStyle
    from pipeline.b3_bridge import classify_text

    # B6: template stability -- same facts, check label/confidence agree
    for style in TemplateStyle:
        result = synthesize_message(cluster, b2_result, context="urban", template=style)
        verdict = classify_text(result["text"], result)
        print(style.name, verdict["label"], verdict["confidence"], verdict["risk_level"])
    # Compare: do all 3 templates agree on label? How much does confidence vary?
    # Report standard deviation of confidence across templates per scenario.

    # B3: adversarial blind-spot -- construct a structured input engineered
    # to produce synthesized text that avoids B3's training-distribution
    # phrasing (e.g. paraphrase DENM cause codes into plain English, remove
    # RSU-style prefixes) while keeping the same underlying malicious facts,
    # and check whether B3 still flags it. If confidence drops sharply
    # under paraphrase alone (same facts, different wording only), that is
    # exactly the kind of adversarial surface a reviewer will ask about.
    """)
else:
    print("Real B3 available -- running template-stability check across the 5-message cluster...")
    verdicts = {}
    for style in TemplateStyle:
        result = synthesize_message(cluster, _LEAKY_B2_RESULT, context="urban", template=style)
        verdict = classify_text(result["text"], result)
        verdicts[style.name] = verdict
        print(f"  [{style.name}] label={verdict['label']} confidence={verdict['confidence']} "
              f"risk_level={verdict['risk_level']}")
    labels = {v["label"] for v in verdicts.values()}
    check("B6: all three templates agree on label for the same underlying facts",
          len(labels) == 1, f"labels seen: {labels}")
    confidences = [v["confidence"] for v in verdicts.values() if v["confidence"] is not None]
    if len(confidences) >= 2:
        spread = max(confidences) - min(confidences)
        check("B6: confidence spread across templates is small (<0.15)", spread < 0.15, f"spread={spread:.3f}")

print()
print("=" * 100)
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S): {_FAILURES}")
    sys.exit(1)
print("All GPU-independent B3/B6 checks passed. See above for what still needs a real B3 run.")
sys.exit(0)
