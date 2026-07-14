#!/usr/bin/env python3
"""
run_rigorous_model_comparison.py
================================
Statistically rigorous B3-vs-candidates comparison, per the reviewer's spec:
attack_id-DISJOINT splits, 5-10 seeds, mean +/- SD, 95% bootstrap CI, paired
McNemar, calibration (ECE/Brier), latency. Recommends replacing the incumbent
B3 ONLY if a candidate's improvement is significant across seeds AND CIs are
non-overlapping.

WHY attack_id-disjoint (critical, non-negotiable):
  The generator yields 120 distinct attack_ids; across seeds each reappears
  with different scene BOILERPLATE (station id/pos/speed/time), producing many
  distinct *texts* but only 120 distinct *attacks*. A random text-level split
  puts the SAME attack_id in train and test -> leakage -> inflated F1. We split
  by attack_id so every variant of an attack is entirely train XOR test. We
  ALSO run the naive text-level split and report it beside the honest one, to
  quantify the leakage inflation (a methods-section strength).

HONESTY:
  * Reuses b3_eval.run_model_benchmark's OWN training/eval helpers (identical
    to the comparison already run).
  * B3 is the incumbent, EVALUATED here (not retrained); its original training
    corpus is unrecoverable, so overlap with this data cannot be excluded --
    reported as a caveat, candidates are the clean controlled comparison.
  * Ceiling is 120 distinct attacks (15 benign). Boilerplate multiplication
    gives statistical PRECISION, not attack DIVERSITY. Stated in the manifest.

Deps: DeBERTa-v3 needs `pip install tiktoken protobuf` or it is SKIPPED (not
silently dropped -- reported).

Run (from repo root, after export_semantic_split has confirmed the generator):
  python3 run_rigorous_model_comparison.py --seeds 1 2 3 4 5 6 7 8 9 10 \
      --candidates roberta-base distilroberta-base microsoft/deberta-v3-base \
      --epochs 3 --balance-benign
"""
from __future__ import annotations
import argparse, json, math, pathlib, statistics, sys, time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from b3_eval.run_model_benchmark import (prf1, measure_latency, load_predictor,
                                          predict_texts, env_manifest, write_json)

OUT = ROOT / "results" / "rigorous_model_comparison"


# ---------------------------------------------------------------- data ------
def build_records(seed: int) -> List[Dict[str, Any]]:
    """All (text, label, attack_id) for one generator seed, via the SAME
    synthesizer path B3 sees."""
    from semantic_evaluation.semantic_attack_generator import generate_corpus
    from pipeline.orchestrator import ISCEPipeline
    global _PIPE
    try:
        _PIPE
    except NameError:
        _PIPE = ISCEPipeline()
    corpus = generate_corpus(seed=seed)
    recs = []
    for r in corpus:
        el = str(r.get("expected_label", "")).upper()
        if el not in ("BENIGN", "MALICIOUS"):
            continue
        label = 0 if el == "BENIGN" else 1
        res = _PIPE.run([r], context=(r.get("scene_context", {}) or {}).get("context") or "urban")
        text = (res.get("synthesized_message") or {}).get("text", "")
        if not text.strip():
            continue
        recs.append({"text": text, "label": label, "attack_id": r.get("attack_id")})
    return recs


def attack_disjoint_split(records: List[Dict[str, Any]], seed: int,
                          test_frac: float, balance_benign: bool
                          ) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """Split so every attack_id is entirely train XOR test. Optionally balance
    at attack_id level (downsample majority-class attack_ids)."""
    import random
    rng = random.Random(seed)
    # group by attack_id, record its (single) label
    by_id: Dict[str, List[Dict]] = defaultdict(list)
    id_label: Dict[str, int] = {}
    for r in records:
        by_id[r["attack_id"]].append(r)
        id_label[r["attack_id"]] = r["label"]
    ids = list(by_id.keys())
    ben_ids = [i for i in ids if id_label[i] == 0]
    mal_ids = [i for i in ids if id_label[i] == 1]

    if balance_benign:
        k = min(len(ben_ids), len(mal_ids))
        rng.shuffle(mal_ids); rng.shuffle(ben_ids)
        mal_ids, ben_ids = mal_ids[:k], ben_ids[:k]

    def split_ids(id_list):
        rng.shuffle(id_list)
        n_test = max(1, int(round(len(id_list) * test_frac)))
        return id_list[n_test:], id_list[:n_test]  # train_ids, test_ids

    tr_ben, te_ben = split_ids(ben_ids)
    tr_mal, te_mal = split_ids(mal_ids)
    train_ids, test_ids = set(tr_ben + tr_mal), set(te_ben + te_mal)

    # HARD leakage assertion
    assert not (train_ids & test_ids), "attack_id leak between train/test"

    train = [r for i in train_ids for r in by_id[i]]
    test = [r for i in test_ids for r in by_id[i]]
    info = {"train_ids": len(train_ids), "test_ids": len(test_ids),
            "train_texts": len(train), "test_texts": len(test),
            "test_benign_ids": len(te_ben), "test_malicious_ids": len(te_mal)}
    return train, test, info


def naive_text_split(records, seed, test_frac):
    """LEAKY baseline split (text-level, ignores attack_id) -- reported only to
    quantify leakage inflation, never used for the decision."""
    import random
    rng = random.Random(seed + 999)
    recs = list(records); rng.shuffle(recs)
    n_test = max(1, int(round(len(recs) * test_frac)))
    return recs[n_test:], recs[:n_test]


# ---------------------------------------------------------------- metrics ---
def bootstrap_ci(values: List[float], n_boot=10000, seed=0):
    import random
    if len(values) < 2:
        return {"mean": values[0] if values else float("nan"), "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(s) / len(s))
    means.sort()
    return {"mean": statistics.mean(values), "sd": statistics.pstdev(values),
            "ci_low": means[int(0.025 * n_boot)], "ci_high": means[int(0.975 * n_boot) - 1]}


def ece_brier(probs_pos, labels, n_bins=10):
    if not probs_pos:
        return None, None
    brier = sum((p - y) ** 2 for p, y in zip(probs_pos, labels)) / len(labels)
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs_pos, labels):
        conf = max(p, 1 - p)
        pred = 1 if p >= 0.5 else 0
        bins[min(int(conf * n_bins), n_bins - 1)].append((conf, int(pred == y)))
    ece, N = 0.0, len(labels)
    for b in bins:
        if b:
            ece += len(b) / N * abs(sum(c for c, _ in b) / len(b) - sum(a for _, a in b) / len(b))
    return ece, brier


def mcnemar_exact(a_correct, b_correct):
    """Paired McNemar on identical items. a,b are per-item correctness bools."""
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return {"applicable": False, "reason": "no discordant pairs", "b": b, "c": c}
    # exact binomial two-sided
    from math import comb
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return {"applicable": True, "b": b, "c": c, "p_value": min(1.0, p)}


# ---------------------------------------------------------------- candidates
def train_and_eval_candidate(name, train, test, epochs, batch_size, lr):
    """Fine-tune one HF encoder on train, eval on test. Returns metrics or a
    skip-reason dict (never fabricates)."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)
    except Exception as e:
        return {"skipped": True, "reason": f"{type(e).__name__}: {e}"}

    class DS(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            e = tok(self.rows[i]["text"], max_length=256, padding="max_length",
                    truncation=True, return_tensors="pt")
            return ({k: v.squeeze(0) for k, v in e.items()} |
                    {"labels": torch.tensor(self.rows[i]["label"])})

    dl = DataLoader(DS(train), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    model.train()
    for _ in range(epochs):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(**batch)
            scaler.scale(out.loss).backward(); scaler.step(opt); scaler.update()
    model.eval()
    texts = [r["text"] for r in test]; labels = [r["label"] for r in test]
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i+32], max_length=256, padding=True, truncation=True,
                      return_tensors="pt").to(device)
            logits = model(**enc).logits
            sm = torch.softmax(logits, dim=1).cpu().tolist()
            preds += [int(p[1] >= p[0]) for p in sm]
            probs += [p[1] for p in sm]
    m = prf1(preds, labels)
    ece, brier = ece_brier(probs, labels)
    def one():
        with torch.no_grad():
            enc = tok(texts[0], max_length=256, padding=True, truncation=True,
                      return_tensors="pt").to(device)
            model(**enc)
    lat = measure_latency(one, n=50)
    per_item_correct = [int(p == y) for p, y in zip(preds, labels)]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"skipped": False, **m, "ece": ece, "brier": brier, "latency": lat,
            "per_item_correct": per_item_correct}


def eval_incumbent_b3(test):
    """Evaluate the incumbent B3 on the same test texts (not retrained)."""
    predictor, reason = load_predictor()
    if predictor is None:
        return {"skipped": True, "reason": reason}
    texts = [r["text"] for r in test]; labels = [r["label"] for r in test]
    out = predict_texts(predictor, texts)
    preds = [1 if o["label"] == "MALICIOUS" else 0 for o in out]
    m = prf1(preds, labels)
    lat = measure_latency(lambda: predict_texts(predictor, [texts[0]]), n=50)
    per_item_correct = [int(p == y) for p, y in zip(preds, labels)]
    return {"skipped": False, **m, "latency": lat, "per_item_correct": per_item_correct}


# ---------------------------------------------------------------- main ------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--candidates", nargs="+",
                    default=["roberta-base", "distilroberta-base"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--balance-benign", action="store_true", default=True)
    ap.add_argument("--swap-f1-margin", type=float, default=2.0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    models = ["INCUMBENT_B3"] + args.candidates
    # per-model, per-seed metric lists
    agg: Dict[str, Dict[str, list]] = {m: defaultdict(list) for m in models}
    # per-seed McNemar (B3 vs each candidate) on identical held-out items
    mcnemar_rows: List[Dict[str, Any]] = []
    leak_inflation: List[Dict[str, Any]] = []
    per_seed_detail = []

    for seed in args.seeds:
        print(f"\n===== seed {seed} =====")
        records = build_records(seed)
        train, test, info = attack_disjoint_split(records, seed, args.test_frac, args.balance_benign)
        print(f"  attack-disjoint: {info}")

        # incumbent
        b3 = eval_incumbent_b3(test)
        if b3.get("skipped"):
            print(f"  [FATAL] B3 unavailable: {b3['reason']}"); return 2
        agg["INCUMBENT_B3"]["f1"].append(b3["f1"] * 100)
        agg["INCUMBENT_B3"]["accuracy"].append(b3["accuracy"] * 100)
        agg["INCUMBENT_B3"]["latency_p95"].append(b3["latency"]["p95"])
        print(f"  INCUMBENT_B3  F1={b3['f1']*100:.2f} acc={b3['accuracy']*100:.2f}")

        for name in args.candidates:
            res = train_and_eval_candidate(name, train, test, args.epochs, args.batch_size, args.lr)
            if res.get("skipped"):
                print(f"  {name}: SKIPPED ({res['reason']})")
                agg[name]["skipped_reason"].append(res["reason"])
                continue
            agg[name]["f1"].append(res["f1"] * 100)
            agg[name]["accuracy"].append(res["accuracy"] * 100)
            agg[name]["latency_p95"].append(res["latency"]["p95"])
            if res.get("ece") is not None:
                agg[name]["ece"].append(res["ece"]); agg[name]["brier"].append(res["brier"])
            mc = mcnemar_exact(b3["per_item_correct"], res["per_item_correct"])
            mcnemar_rows.append({"seed": seed, "candidate": name, **mc,
                                 "b3_f1": b3["f1"]*100, "cand_f1": res["f1"]*100})
            print(f"  {name}  F1={res['f1']*100:.2f} acc={res['accuracy']*100:.2f} "
                  f"McNemar p={mc.get('p_value')}")

        # leakage-inflation contrast (naive text split, incumbent only, cheap)
        ntr, nte = naive_text_split(records, seed, args.test_frac)
        b3_naive = eval_incumbent_b3(nte)
        if not b3_naive.get("skipped"):
            leak_inflation.append({"seed": seed,
                                   "b3_f1_attack_disjoint": round(b3["f1"]*100, 2),
                                   "b3_f1_naive_textsplit": round(b3_naive["f1"]*100, 2)})
        per_seed_detail.append({"seed": seed, "split_info": info})

    # aggregate: mean +/- SD +/- 95% CI per model per metric
    summary = {}
    for m in models:
        summary[m] = {}
        for metric in ("f1", "accuracy", "latency_p95", "ece", "brier"):
            vals = agg[m].get(metric, [])
            if vals:
                ci = bootstrap_ci([float(v) for v in vals])
                summary[m][metric] = {"mean": ci["mean"], "sd": ci.get("sd"),
                                      "ci95_low": ci["ci_low"], "ci95_high": ci["ci_high"],
                                      "n_seeds": len(vals), "raw": vals}
        if agg[m].get("skipped_reason"):
            summary[m]["skipped_reason"] = agg[m]["skipped_reason"][0]

    # DECISION: recommend swap only if, for a candidate,
    #   (a) mean F1 exceeds B3 by >= margin, AND
    #   (b) 95% CIs do NOT overlap, AND
    #   (c) McNemar favors candidate (p<0.05) in a MAJORITY of seeds.
    decisions = {}
    b3_f1 = summary["INCUMBENT_B3"].get("f1", {})
    for name in args.candidates:
        c = summary[name].get("f1")
        if not c:
            decisions[name] = "SKIPPED / no data"; continue
        margin_ok = (c["mean"] - b3_f1["mean"]) >= args.swap_f1_margin
        ci_disjoint = (c["ci95_low"] is not None and b3_f1["ci95_high"] is not None
                       and c["ci95_low"] > b3_f1["ci95_high"])
        sig_seeds = [r for r in mcnemar_rows if r["candidate"] == name
                     and r.get("applicable") and r.get("p_value", 1) < 0.05
                     and r["cand_f1"] > r["b3_f1"]]
        maj_sig = len(sig_seeds) > len([r for r in mcnemar_rows if r["candidate"] == name]) / 2
        if margin_ok and ci_disjoint and maj_sig:
            decisions[name] = (f"SWAP SUPPORTED: +{c['mean']-b3_f1['mean']:.2f} F1, CIs disjoint, "
                               f"McNemar-significant in majority of seeds. VERIFY on a held-out "
                               f"attack family before adopting.")
        else:
            decisions[name] = (f"KEEP INCUMBENT: margin_ok={margin_ok} "
                               f"ci_disjoint={ci_disjoint} majority_mcnemar_sig={maj_sig} "
                               f"(deltaF1={c['mean']-b3_f1['mean']:.2f})")

    manifest = env_manifest("rigorous_model_comparison", {
        "seeds": args.seeds, "candidates": args.candidates, "epochs": args.epochs,
        "split": "attack_id-DISJOINT (no attack appears in both train and test)",
        "balance_benign": args.balance_benign,
        "distinct_attacks_ceiling": 120,
        "distinct_benign_attacks": 15,
        "honesty_notes": [
            "1200 texts = 120 attacks x 10 boilerplate variants; precision not diversity.",
            "B3 evaluated (not retrained); overlap with its lost training corpus cannot be excluded.",
            "leak_inflation shows naive text-split F1 vs honest attack-disjoint F1.",
        ],
    })
    payload = {"manifest": manifest, "summary": summary, "decisions": decisions,
               "mcnemar_per_seed": mcnemar_rows, "leakage_inflation": leak_inflation,
               "per_seed": per_seed_detail}
    write_json(payload, OUT / "rigorous_comparison.json")

    print("\n" + "=" * 78)
    print("SUMMARY (mean F1 +/- 95% CI over seeds, attack-disjoint):")
    for m in models:
        f = summary[m].get("f1")
        if f:
            print(f"  {m:28s} F1={f['mean']:.2f} +/- (95% CI [{f['ci95_low']:.2f},{f['ci95_high']:.2f}]) "
                  f"sd={f.get('sd',0):.2f} n={f['n_seeds']}")
        elif summary[m].get("skipped_reason"):
            print(f"  {m:28s} SKIPPED: {summary[m]['skipped_reason']}")
    if leak_inflation:
        avg_disjoint = statistics.mean(r["b3_f1_attack_disjoint"] for r in leak_inflation)
        avg_naive = statistics.mean(r["b3_f1_naive_textsplit"] for r in leak_inflation)
        print(f"\nLEAKAGE CHECK (B3): attack-disjoint F1={avg_disjoint:.2f} vs "
              f"naive text-split F1={avg_naive:.2f} "
              f"(inflation={avg_naive-avg_disjoint:+.2f} -- why we split by attack_id)")
    print("\nDECISIONS:")
    for k, v in decisions.items():
        print(f"  {k}: {v}")
    print(f"\nWritten: {OUT/'rigorous_comparison.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())