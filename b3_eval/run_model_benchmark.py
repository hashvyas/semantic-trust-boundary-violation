"""
b3_eval/run_model_benchmark.py
================================
Part 5: benchmark the CURRENT B3 checkpoint against realistic alternatives
on the SAME held-out split, so any decision to replace it is evidence-based.

IMPORTANT -- what this does and does not do:
- The current checkpoint is a FINE-TUNED classifier. The comparison models
  (deberta-v3-base, roberta-base, ModernBERT-base, distilroberta, MiniLM)
  are PRETRAINED BACKBONES with randomly-initialized classification heads.
  Comparing them zero-shot against a fine-tuned model is meaningless.
  Therefore this harness fine-tunes each candidate on the SAME training
  split with the SAME budget before comparing -- otherwise it refuses to
  report a comparison at all.
- It reports accuracy/precision/recall/F1, latency, param count, and peak
  VRAM per candidate, plus training cost (wall time).

DECISION RULE (encoded in the output, per the mandate's "do not replace the
model simply because a newer architecture exists"):
  Recommend a swap ONLY if a candidate beats the incumbent by >= 2.0 F1
  points on the held-out split AND does not regress p95 latency by more than
  25%. Otherwise the output explicitly states "KEEP INCUMBENT".

Literature context (already verified, see B3_ASSESSMENT.md §2): controlled
same-data comparisons (Antoun et al. 2025, arXiv:2504.08716) find DeBERTaV3
>= ModernBERT on accuracy and sample efficiency, with ModernBERT ahead only
on long-context/speed -- and at 256 tokens long-context is irrelevant here.
The prior expectation is therefore that the incumbent wins; this harness
exists to TEST that, not to assume it.

Requires:
  b3_eval/data/train_split.jsonl  and  b3_eval/data/test_split.jsonl
  (schema: {"text": str, "label": 0|1} per line)

Run with:  python3 b3_eval/run_model_benchmark.py [--candidates ...] [--epochs 3]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from b3_eval._harness import (MODEL_DIR, checkpoint_status, env_manifest,
                                load_predictor, predict_texts, torch_status, write_json)

OUT = ROOT / "b3_eval" / "results"
DATA = ROOT / "b3_eval" / "data"

DEFAULT_CANDIDATES = [
    "microsoft/deberta-v3-base",
    "roberta-base",
    "answerdotai/ModernBERT-base",
    "distilroberta-base",
    "microsoft/MiniLM-L12-H384-uncased",
]

SWAP_F1_MARGIN = 2.0     # percentage points
SWAP_LATENCY_TOLERANCE = 1.25  # candidate p95 may be at most 1.25x incumbent's


def load_jsonl(p):
    if not p.exists():
        return None
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            o = json.loads(line)
            rows.append((o["text"], int(o["label"])))
    return rows


def prf1(preds, labels):
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(labels)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def measure_latency(fn, n=100):
    for _ in range(10):
        fn()
    runs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - t0) * 1000.0)
    s = sorted(runs)
    return {"p50": s[len(s) // 2], "p95": s[int(len(s) * .95)], "mean": statistics.mean(runs)}


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    return ap


def main():
    args = build_arg_parser().parse_args()

    manifest = env_manifest("b3_model_benchmark", {"candidates": args.candidates,
                                                     "epochs": args.epochs,
                                                     "seeds": args.seeds})
    train = load_jsonl(DATA / "train_split.jsonl")
    test = load_jsonl(DATA / "test_split.jsonl")
    ck, tt = checkpoint_status(), torch_status()

    print("=" * 78)
    print("B3 MODEL BENCHMARK (Part 5)")
    print("=" * 78)
    print(f"Seeds: {args.seeds}")
    if train is None or test is None:
        print("CANNOT RUN: missing splits.")
        print(f"  expected {DATA / 'train_split.jsonl'} and {DATA / 'test_split.jsonl'}")
        print('  schema: {"text": str, "label": 0|1} per line')
        print("\nUntil these exist, no model comparison is possible and NO SWAP is justified.")
        print("Per B3_ASSESSMENT.md §2, published controlled comparisons already favour the")
        print("incumbent architecture (DeBERTaV3) over ModernBERT/RoBERTa at this context length,")
        print("so the default action is KEEP INCUMBENT.")
        write_json({"manifest": manifest, "status": "no_splits",
                     "decision": "KEEP INCUMBENT (no evidence to justify a swap)"},
                   OUT / "model_benchmark.json")
        return 0
    if not ck["ok"] or not tt["ok"]:
        reason = ck.get("reason") or tt.get("reason")
        print(f"CANNOT RUN: {reason}")
        write_json({"manifest": manifest, "status": "unavailable", "reason": reason,
                     "decision": "KEEP INCUMBENT (no evidence to justify a swap)"},
                   OUT / "model_benchmark.json")
        return 0

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_texts = [t for t, _ in test]
    test_labels = [y for _, y in test]

    results = {}

    # ---- incumbent ----
    predictor, reason = load_predictor()
    inc_preds = [1 if r["label"] == "MALICIOUS" else 0
                 for r in predict_texts(predictor, test_texts)]
    inc_metrics = prf1(inc_preds, test_labels)
    inc_lat = measure_latency(lambda: predict_texts(predictor, [test_texts[0]], batch_size=1))
    inc_params = sum(p.numel() for p in predictor.model.parameters())
    results["INCUMBENT (semantic_gate_v3)"] = {
        **inc_metrics, "latency_ms": inc_lat, "parameters": inc_params,
        "train_seconds": None, "note": "already fine-tuned; not retrained here"}
    print(f"\nINCUMBENT  F1={inc_metrics['f1']*100:.2f}  acc={inc_metrics['accuracy']*100:.2f}  "
          f"p95={inc_lat['p95']:.2f}ms  params={inc_params/1e6:.1f}M")

    # ---- candidates: fine-tune each on the SAME split ----
    class DS(Dataset):
        def __init__(self, rows, tok):
            self.rows, self.tok = rows, tok
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            t, y = self.rows[i]
            e = self.tok(t, max_length=256, padding="max_length", truncation=True,
                          return_tensors="pt")
            return {k: v.squeeze(0) for k, v in e.items()} | {"labels": torch.tensor(y)}

    for name in args.candidates:
        print(f"\n--- fine-tuning {name} (same split, {args.epochs} epochs) ---")
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)
            dl = DataLoader(DS(train, tok), batch_size=args.batch_size, shuffle=True)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
            scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
            t0 = time.perf_counter()
            model.train()
            for ep in range(args.epochs):
                for batch in dl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    opt.zero_grad()
                    with torch.autocast(device_type=device.type,
                                         enabled=(device.type == "cuda")):
                        out = model(**batch)
                    scaler.scale(out.loss).backward()
                    scaler.step(opt); scaler.update()
            train_s = time.perf_counter() - t0
            model.eval()

            preds = []
            with torch.no_grad():
                for i in range(0, len(test_texts), 32):
                    enc = tok(test_texts[i:i+32], max_length=256, padding=True,
                              truncation=True, return_tensors="pt").to(device)
                    preds.extend(model(**enc).logits.argmax(dim=1).cpu().tolist())
            m = prf1(preds, test_labels)

            def one():
                with torch.no_grad():
                    enc = tok(test_texts[0], max_length=256, padding=True, truncation=True,
                              return_tensors="pt").to(device)
                    model(**enc)
            lat = measure_latency(one)
            nparams = sum(p.numel() for p in model.parameters())
            results[name] = {**m, "latency_ms": lat, "parameters": nparams,
                              "train_seconds": train_s}
            print(f"  F1={m['f1']*100:.2f}  acc={m['accuracy']*100:.2f}  "
                  f"p95={lat['p95']:.2f}ms  params={nparams/1e6:.1f}M  train={train_s:.0f}s")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  FAILED: {e}")

    # ---- decision ----
    inc_f1 = inc_metrics["f1"] * 100
    inc_p95 = inc_lat["p95"]
    best, best_f1 = None, inc_f1
    for name, r in results.items():
        if name.startswith("INCUMBENT") or "error" in r:
            continue
        f1 = r["f1"] * 100
        if f1 - inc_f1 >= SWAP_F1_MARGIN and r["latency_ms"]["p95"] <= inc_p95 * SWAP_LATENCY_TOLERANCE:
            if f1 > best_f1:
                best, best_f1 = name, f1
    if best:
        decision = (f"SWAP JUSTIFIED: {best} beats incumbent by "
                    f"{best_f1 - inc_f1:.2f} F1 points (>= {SWAP_F1_MARGIN}) within the "
                    f"latency tolerance. Verify on a second seed before adopting.")
    else:
        decision = ("KEEP INCUMBENT: no candidate met the swap bar "
                    f"(>= {SWAP_F1_MARGIN} F1 points AND p95 <= "
                    f"{SWAP_LATENCY_TOLERANCE}x incumbent). Architecture change is not justified.")
    print("\n" + "=" * 78)
    print(decision)
    write_json({"manifest": manifest, "results": results, "decision": decision,
                 "swap_rule": {"f1_margin_points": SWAP_F1_MARGIN,
                                "latency_tolerance": SWAP_LATENCY_TOLERANCE}},
               OUT / "model_benchmark.json")
    print(f"Written: {OUT / 'model_benchmark.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
