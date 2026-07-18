import json, pathlib, sys
sys.path.insert(0, '.')
from pipeline.b3_bridge import classify_text, preload_classifier

preload_classifier()
msgs = [json.loads(f.read_text()) for f in sorted(pathlib.Path('scenarios/semantic').glob('*.json'))]

for i, msg in enumerate(msgs[:5]):
    text = msg.get('semantic_text', '')
    result = classify_text(text)
    print(f"msg_{i:03d}: label={result.get('label')} conf={result.get('confidence'):.4f} risk={result.get('risk_level')}")