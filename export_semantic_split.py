#!/usr/bin/env python3
"""
export_semantic_split.py  (v4 - synthesized text + optional class balancing)
============================================================================
Writes run_model_benchmark.py's splits from the repo's OWN generator+synthesizer.
NOT B3's original corpus (that is unrecoverable). See split_manifest.json.

v4 adds --balance: downsample the majority class so the benchmark is not
dominated by class imbalance. Without balancing, a trivial all-MALICIOUS
predictor scores F1=0.933 on the native 87.5%-malicious corpus, which makes
the model comparison meaningless (verified: 4 different architectures all
scored an identical 93.33). --balance is ON by default for this reason.

Run:
    python3 export_semantic_split.py --seed 20260713 --test-frac 0.2            # balanced (default)
    python3 export_semantic_split.py --no-balance                               # native imbalance
    python3 b3_eval/run_model_benchmark.py
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, random, sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def map_label(x: Any) -> Optional[int]:
    if not isinstance(x, str): return None
    s = x.strip().upper()
    return 0 if s == "BENIGN" else 1 if s == "MALICIOUS" else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--balance", dest="balance", action="store_true", default=True,
                    help="downsample majority class (default ON)")
    ap.add_argument("--no-balance", dest="balance", action="store_false")
    ap.add_argument("--repo-root", default=str(pathlib.Path(__file__).resolve().parent))
    args = ap.parse_args()

    repo = pathlib.Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo))
    try:
        from semantic_evaluation.semantic_attack_generator import generate_corpus
        from pipeline.orchestrator import ISCEPipeline
    except Exception as e:
        print(f"[FATAL] import failed: {e}"); return 2

    corpus = generate_corpus(seed=args.seed)
    if not isinstance(corpus, list) or not corpus:
        print("[FATAL] generate_corpus empty/unexpected."); return 3
    print("expected_label distribution:", dict(Counter(str(r.get("expected_label")) for r in corpus)))

    pipe = ISCEPipeline()
    pairs: List[Tuple[str, int]] = []
    skipped_label = skipped_notext = 0
    errors: List[str] = []
    for i, rec in enumerate(corpus):
        lab = map_label(rec.get("expected_label"))
        if lab is None: skipped_label += 1; continue
        try:
            result = pipe.run([rec], context=(rec.get("scene_context", {}) or {}).get("context") or "urban")
            text = (result.get("synthesized_message") or {}).get("text", "")
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}"); continue
        if not text or not text.strip(): skipped_notext += 1; continue
        pairs.append((text, lab))
        if (i + 1) % 20 == 0: print(f"  ...synthesized {i+1}/{len(corpus)}")

    if not pairs:
        print("[FATAL] no (text,label) pairs."); 
        if errors: print("  first errors:", errors[:3])
        return 4

    # dedup exact texts
    seen, deduped = set(), []
    for t, l in pairs:
        h = hashlib.sha256(t.encode()).hexdigest()
        if h in seen: continue
        seen.add(h); deduped.append((t, l))
    n_dups = len(pairs) - len(deduped)

    rng = random.Random(args.seed)
    by = {0: [t for t, l in deduped if l == 0], 1: [t for t, l in deduped if l == 1]}
    native = {0: len(by[0]), 1: len(by[1])}

    n_downsampled = 0
    if args.balance:
        k = min(len(by[0]), len(by[1]))
        for l in (0, 1):
            if len(by[l]) > k:
                rng.shuffle(by[l]); n_downsampled += len(by[l]) - k; by[l] = by[l][:k]

    # stratified split
    train, test = [], []
    for l in (0, 1):
        items = [(t, l) for t in by[l]]
        rng.shuffle(items)
        nt = int(round(len(items) * args.test_frac))
        test += items[:nt]; train += items[nt:]
    rng.shuffle(train); rng.shuffle(test)

    out_dir = repo / "b3_eval" / "data"; out_dir.mkdir(parents=True, exist_ok=True)
    def wj(p, rows):
        with open(p, "w") as f:
            for t, l in rows: f.write(json.dumps({"text": t, "label": l}) + "\n")
    wj(out_dir / "train_split.jsonl", train); wj(out_dir / "test_split.jsonl", test)
    def dist(rows): return {"n": len(rows), "malicious": sum(l for _, l in rows),
                            "benign": sum(1 for _, l in rows if l == 0)}

    manifest = {
        "artifact": "REGENERATED benchmark split for run_model_benchmark.py",
        "IS_NOT": "B3's original training corpus (unrecoverable; verified by exhaustive search).",
        "text_source": "pipeline synthesized scene text (result['synthesized_message']['text']) "
                        "-- same text B3 classifies (semantic_attack_evaluation.py L138).",
        "generated_by": "semantic_attack_generator.generate_corpus + ISCEPipeline synthesizer",
        "seed": args.seed, "test_frac": args.test_frac,
        "class_balancing": {
            "enabled": args.balance,
            "native_counts": native,
            "majority_downsampled_by": n_downsampled,
            "rationale": "Native corpus is 87.5% malicious; without balancing a trivial "
                         "all-MALICIOUS predictor scores F1=0.933, making model comparison "
                         "uninformative. Balancing makes the benchmark discriminative.",
        },
        "exact_duplicate_texts_removed": n_dups,
        "records_skipped_ambiguous_label": skipped_label,
        "records_skipped_empty_text": skipped_notext,
        "pipeline_errors": len(errors),
        "train": dist(train), "test": dist(test),
        "train_sha256_16": hashlib.sha256((out_dir/"train_split.jsonl").read_bytes()).hexdigest()[:16],
        "test_sha256_16": hashlib.sha256((out_dir/"test_split.jsonl").read_bytes()).hexdigest()[:16],
        "paper_caveats": [
            "Newly generated split; NOT B3's training data.",
            "Balanced split: report this as the discriminative comparison.",
            "Report freshly-trained comparison models as the PRIMARY controlled comparison.",
            "Report B3 WITH the caveat that overlap with its unavailable training corpus "
            "cannot be excluded.",
            "Test set is small; report exact n and treat single-point metrics as high-variance. "
            "Prefer multi-seed CIs.",
        ],
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote (balance={args.balance}):")
    print(f"  train {dist(train)}")
    print(f"  test  {dist(test)}")
    print(f"  {out_dir/'split_manifest.json'}")
    if n_downsampled: print(f"[note] downsampled majority class by {n_downsampled} to balance.")
    if n_dups: print(f"[note] removed {n_dups} duplicate text(s).")
    if skipped_label: print(f"[WARN] skipped {skipped_label} ambiguous-label record(s).")
    if skipped_notext: print(f"[WARN] skipped {skipped_notext} empty-text record(s).")
    if errors: print(f"[WARN] {len(errors)} pipeline error(s); first: {errors[0]}")
    print("\nWARNING: even balanced, the test set is small (n<=~24). Report exact n and use "
          "multi-seed CIs; do not over-read a single-run F1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())