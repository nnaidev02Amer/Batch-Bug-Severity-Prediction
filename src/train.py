"""Train, compare, calibrate and save the severity classifier.

Usage:
    python -m src.train                    # original 99 rows + collected Bugzilla reports (default)
    python -m src.train --original-only    # previous protocol: original 99 rows only
    python -m src.train --data my.csv      # any labelled CSV (random stratified split)

Method (leakage-safe):
1. Fixed held-out test sets (see src/data.py): ``orig_test`` = the same 20 original
   rows the previous version held out; ``ext_test`` = 20 % of the collected reports
   (group split, near-duplicates never straddle train/test). A custom --data CSV
   gets a stratified random split instead.
2. Candidates (4 classifiers x feature sets) are scored with stratified, group-aware
   k-fold CV on the TRAINING pool only; the vectoriser is fitted inside each fold.
3. The best candidate by CV macro-F1 (not accuracy) is wrapped in probability
   calibration, fitted on the training pool and evaluated once on each test set.
4. For deployment the same recipe is re-fitted on ALL rows. Reported test metrics are
   those of step 3 (they estimate, not measure, the deployed model).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from .evaluation import evaluate_predictions, majority_baseline
from .modeling import (CLASSIFIERS, FEATURE_DESCRIPTIONS, align_proba, calibrated, cv_score, decide,
                       make_cv_splits, make_pipeline, probability_metrics, risk_coverage)
from .preprocessing import load_training_data
from .utils import (CLASSES, DEFAULT_DATA, DEFAULT_MODEL, EXTRA_DATA, REPORT_PATH,
                    BugSeverityError)

# Defaults for a NEW training run (experiments: reports/experiments.json, stages s2/s6;
# chosen from cross-validation on the training pool only, never from the test sets).
# NOTE: the model shipped in models/ was produced by the previous version of this
# script (original 99 rows, no calibration); a re-run produces a different model.
DEFAULT_FEATURES = ("word", "char", "word+char")
DEFAULT_NORMALIZER = "technical"  # all three normalisers tied in CV (s2); keep the existing one
DEFAULT_CALIBRATION = "sigmoid"
DEFAULT_DECISION = "argmax"  # "prior_corrected" trades macro-F1 for Non-Critical recall (s6); not adopted yet
DEFAULT_THRESHOLD = 0.6

PREVIOUS_RESULT = {  # previous version (MODEL_REPORT.md of 2026-11-25T06:22:10Z), kept for comparison
    "algorithm": "Random Forest", "train_rows": 79, "test_rows": 20, "accuracy": 0.800,
    "macro_f1": 0.563, "weighted_f1": 0.760, "noncritical_recall": 0.0,
    "confusion_matrix": [[6, 2, 0], [0, 10, 0], [0, 2, 0]]}


def candidates(seed: int, features: str = "word+char", normalizer: str = "technical"):
    """The four compared classifiers on one feature set (kept for API compatibility)."""
    return {m: make_pipeline(m, features, normalizer, seed) for m in CLASSIFIERS}


def _evaluate(model, priors, X, y) -> dict:
    """Metrics of a fitted model on one test set, plus probability quality."""
    proba = align_proba(model, model.predict_proba(X)) if hasattr(model, "predict_proba") else None
    pred = decide(proba, priors) if (proba is not None and priors) else model.predict(X)
    m = evaluate_predictions(list(y), list(pred))
    if proba is not None:
        m["probability"] = probability_metrics(y, proba)
        m["risk_coverage"] = risk_coverage(y, proba, pred)
    return m


def train(df, seed: int = 42, test_size: float = 0.2, cv_splits: int = 5, cv_repeats: int = 3, *,
          test_sets: dict | None = None, groups=None, features=("word+char",),
          normalizer: str = "technical", calibration: str | None = DEFAULT_CALIBRATION,
          decision: str = "argmax", confidence_threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Run the comparison and return {pipeline, metadata}.

    ``df`` (columns text, label[, raw_severity]) is the TRAINING pool when
    ``test_sets`` ({name: DataFrame}) is given; otherwise it is split stratified
    80/20 here (the previous behaviour). ``groups`` (aligned with df) makes CV
    group-aware. The first test set is reported as the primary ``test_metrics``.
    """
    counts = df["label"].value_counts()
    missing = [c for c in CLASSES if c not in counts.index]
    if missing:
        raise BugSeverityError(f"Training data has no examples for class(es): {missing}")
    if counts.min() < 4:
        raise BugSeverityError(
            f"Need at least 4 examples per class for stratified evaluation; got {counts.to_dict()}")
    if decision not in ("argmax", "prior_corrected"):
        raise ValueError(f"Unknown decision rule {decision!r}")

    if test_sets is None:
        idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=test_size,
                                          stratify=df["label"], random_state=seed)
        test_sets = {"test": df.iloc[idx_te]}
        if groups is not None:
            groups = np.asarray(groups)[idx_tr]
        df = df.iloc[idx_tr]
    X_tr, y_tr = df["text"].to_numpy(), df["label"].to_numpy()
    splits = make_cv_splits(y_tr, groups, cv_splits, cv_repeats, seed)

    comparison = {}
    for feat in features:
        for name, pipe in candidates(seed, feat, normalizer).items():
            key = name if len(features) == 1 else f"{name} | {feat}"
            comparison[key] = cv_score(pipe, X_tr, y_tr, splits) | {"features": feat}
            r = comparison[key]
            print(f"  {key:40s} CV macro-F1 {r['cv_macro_f1_mean']:.3f} ± {r['cv_macro_f1_std']:.3f}"
                  f"  acc {r['cv_accuracy_mean']:.3f}  NC-recall {r['cv_noncritical_recall_mean']:.2f}",
                  flush=True)
    best_key = max(comparison, key=lambda n: comparison[n]["cv_macro_f1_mean"])
    best_model = best_key.split(" | ")[0]
    best_feat = comparison[best_key]["features"]

    def recipe():
        pipe = make_pipeline(best_model, best_feat, normalizer, seed)
        return calibrated(pipe, calibration, seed) if calibration else pipe

    priors = ({c: float((y_tr == c).mean()) for c in CLASSES}
              if decision == "prior_corrected" else None)

    eval_model = recipe().fit(X_tr, y_tr)
    test_results = {name: _evaluate(eval_model, priors, t["text"].to_numpy(), t["label"].to_numpy())
                    for name, t in test_sets.items()}
    primary = next(iter(test_sets))
    baseline = majority_baseline(y_tr, test_sets[primary]["label"].to_numpy())

    # Final deployment model: same recipe on ALL rows (training pool + test sets).
    all_df = pd.concat([df, *test_sets.values()], ignore_index=True)
    X_all, y_all = all_df["text"].to_numpy(), all_df["label"].to_numpy()
    final = recipe().fit(X_all, y_all)
    final_priors = ({c: float((y_all == c).mean()) for c in CLASSES}
                    if decision == "prior_corrected" else None)

    raw_col = all_df["raw_severity"] if "raw_severity" in all_df else pd.Series(dtype=str)
    metadata = {
        "algorithm": best_model,
        "feature_extraction": f"{FEATURE_DESCRIPTIONS[best_feat]}; normaliser '{normalizer}' "
                              "(technical tokens preserved)",
        "features": best_feat, "normalizer": normalizer,
        "classes": CLASSES,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "n_total": int(len(all_df)), "n_train": int(len(df)),
        "n_test": int(len(test_sets[primary])),
        "class_counts": {c: int((y_all == c).sum()) for c in CLASSES},
        "raw_label_counts": raw_col.value_counts().to_dict(),
        "cv": {"splits": len(splits) // cv_repeats, "repeats": cv_repeats,
               "group_aware": groups is not None, "on": "training pool only"},
        "model_comparison": comparison,
        "selection_metric": "CV macro-F1",
        "calibration": calibration or "none",
        "decision_rule": decision,
        "decision_priors": final_priors,
        "confidence_threshold": confidence_threshold,
        "primary_test_set": primary,
        "test_metrics": test_results[primary],
        "test_sets": test_results,
        "majority_baseline": baseline,
        "previous_result": PREVIOUS_RESULT,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__},
    }
    return {"pipeline": final, "metadata": metadata}


# ---------------------------------------------------------------- report
def _fmt_table(m: dict) -> list[str]:
    L = ["| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for c in CLASSES:
        p = m["per_class"][c]
        L.append(f"| {c} | {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {p['support']} |")
    L += ["", "Confusion matrix (rows = true, columns = predicted; order " + ", ".join(m["labels"]) + "):",
          "", "```", *[str(r) for r in m["confusion_matrix"]], "```"]
    return L


def write_report(meta: dict, data_desc: list[str], path: Path = REPORT_PATH) -> None:
    """Generate MODEL_REPORT.md from the actual results (no hand-typed numbers)."""
    prev = meta["previous_result"]
    L = ["# Model Report", "",
         f"*Generated automatically by `src/train.py` on {meta['trained_at_utc']} (UTC). "
         "Research experiments behind the design choices: `reports/experiments.json`.*", "",
         "> **Read this first:** all data comes from Mozilla Bugzilla (Core product family, 2006–2008). "
         "The test metrics below estimate performance on *similar* reports only. The model is a triage "
         "aid that a human must confirm; it is not validated for production use on other projects.", "",
         "## Data", "", *data_desc,
         f"* Mapped classes (all rows): {meta['class_counts']}",
         f"* Raw labels (all rows): {meta['raw_label_counts']}",
         "* Mapping (unchanged from the previous version): critical/blocker/major → Critical; normal → Normal; "
         "minor/trivial/enhancement → Non-Critical. Mapping evidence: `reports/experiments.json` (stage s5).", "",
         "## Setup", "",
         f"* Training pool: {meta['n_train']} rows. Candidate selection: {meta['cv']['splits']}-fold × "
         f"{meta['cv']['repeats']} stratified{' group-aware' if meta['cv']['group_aware'] else ''} CV on the "
         "training pool only, vectoriser fitted inside each fold; selection metric **macro-F1** (not accuracy).",
         f"* Features: {meta['feature_extraction']}.",
         "* Class imbalance: `class_weight='balanced'` (LR/SVM/RF) or uniform priors (NB).",
         f"* Probability calibration: {meta['calibration']} (5-fold, out-of-fold scores). Decision rule: "
         f"{meta['decision_rule']}" + (" (argmax of p(class)/prior(class), so calibration does not swamp the "
                                        "minority class)" if meta['decision_rule'] == 'prior_corrected' else "") + ".",
         f"* Confidence threshold: predictions whose class probability is below {meta['confidence_threshold']:.2f} "
         "are shown as **\"Low confidence – Manual review recommended.\"**",
         f"* Versions: {meta['versions']}", "",
         "## Model comparison (CV on the training pool)", "",
         "| Model | CV macro-F1 | CV accuracy | CV weighted-F1 | CV Non-Critical recall |", "|---|---|---|---|---|"]
    for n, r in sorted(meta["model_comparison"].items(), key=lambda kv: -kv[1]["cv_macro_f1_mean"]):
        mark = " **(selected)**" if n.split(" | ")[0] == meta["algorithm"] and r["features"] == meta["features"] else ""
        L.append(f"| {n}{mark} | {r['cv_macro_f1_mean']:.3f} ± {r['cv_macro_f1_std']:.3f} "
                 f"| {r['cv_accuracy_mean']:.3f} ± {r['cv_accuracy_std']:.3f} "
                 f"| {r['cv_weighted_f1_mean']:.3f} | {r['cv_noncritical_recall_mean']:.3f} |")
    b = meta["majority_baseline"]
    L += ["", f"Majority-class baseline on `{meta['primary_test_set']}` (always '{b['class']}'): "
          f"accuracy {b['accuracy']:.3f}, macro-F1 {b['macro_f1']:.3f}.", ""]
    for name, m in meta["test_sets"].items():
        L += [f"## Held-out test `{name}` – {meta['algorithm']} ({m['n_samples']} reports)", "",
              f"* Accuracy {m['accuracy']:.3f} · macro-F1 **{m['macro']['f1']:.3f}** · weighted-F1 {m['weighted']['f1']:.3f}",
              f"* Macro precision / recall: {m['macro']['precision']:.3f} / {m['macro']['recall']:.3f}", ""]
        L += _fmt_table(m)
        if "probability" in m:
            p = m["probability"]
            L += ["", f"Probability quality: log-loss {p['log_loss']:.3f}, Brier {p['brier']:.3f}, "
                      f"expected calibration error {p['ece']:.3f}.", "",
                  "| Confidence threshold | Share answered | Accuracy (answered) | Accuracy (flagged for review) |",
                  "|---|---|---|---|"]
            for r in m["risk_coverage"]:
                a = "–" if r["accuracy_confident"] is None else f"{r['accuracy_confident']:.3f}"
                f = "–" if r["accuracy_flagged"] is None else f"{r['accuracy_flagged']:.3f}"
                L.append(f"| {r['threshold']:.1f} | {r['coverage']:.1%} | {a} | {f} |")
        L.append("")
    L += ["## Comparison with the previous version", "",
          "| | Previous (RF, 79 train rows) | Current |", "|---|---|---|"]
    if "orig_test" in meta["test_sets"]:
        o = meta["test_sets"]["orig_test"]
        L += [f"| Same 20 original test rows – accuracy | {prev['accuracy']:.3f} | {o['accuracy']:.3f} |",
              f"| Same 20 original test rows – macro-F1 | {prev['macro_f1']:.3f} | {o['macro']['f1']:.3f} |",
              f"| Same 20 original test rows – Non-Critical recall | {prev['noncritical_recall']:.0%} (0/2) "
              f"| {o['per_class']['Non-Critical']['recall']:.0%} "
              f"({round(o['per_class']['Non-Critical']['recall'] * o['per_class']['Non-Critical']['support'])}"
              f"/{o['per_class']['Non-Critical']['support']}) |"]
    L += ["", "The previous model's score on the larger `ext_test` set is in reports/experiments.json (stage s3); "
          "20 rows are too few to compare models reliably on their own.", "",
          "## Final deployed model", "",
          f"The selected recipe is re-fitted on **all {meta['n_total']} rows** and saved to `models/trained_model.joblib`. "
          "Its performance is estimated by the held-out numbers above, not measured separately.", "",
          "## Known limitations", "",
          "* One project family (Mozilla Core, 2006–2008), short one-line summaries only; no evidence the model transfers "
          "to other products, trackers or modern writing styles.",
          "* Bugzilla severities are subjective; identical-looking reports carry different labels (see error analysis). "
          "This label noise caps achievable accuracy.",
          "* Non-Critical is still the weakest class (~7 % of data, and a mixture of minor defects, trivial issues and "
          "feature requests).",
          "* Calibrated probabilities describe *average* reliability on similar data; a high confidence is not a "
          "guarantee that an individual prediction is correct.",
          "* Word2Vec (original approach) could not be evaluated: `gensim` fails to import against current SciPy on Python 3.14.",
          "* Predictions are recommendations for a human triager, not determinations of business impact.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="custom labelled CSV (random stratified split)")
    ap.add_argument("--original-only", action="store_true", help="use only the original 99-row sample")
    ap.add_argument("--out", default=str(DEFAULT_MODEL))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--no-report", action="store_true", help="do not rewrite MODEL_REPORT.md")
    args = ap.parse_args(argv)

    opts = dict(seed=args.seed, features=DEFAULT_FEATURES, normalizer=DEFAULT_NORMALIZER, calibration=DEFAULT_CALIBRATION,
                decision=DEFAULT_DECISION, confidence_threshold=args.threshold)
    try:
        if args.data:
            df, info = load_training_data(args.data)
            print(f"Loaded {info['rows_used']} rows from {args.data}; classes: {info['class_counts']}")
            result = train(df, **opts)
            desc = [f"* Source: `{args.data}`; rows used {info['rows_used']} of {info['rows_raw']}; "
                    f"dropped: {info['dropped_duplicates']} duplicates, {info['dropped_conflicting_labels']} conflicting, "
                    f"unknown labels {info['dropped_unknown_label'] or 'none'}. Random stratified 80/20 split."]
        else:
            from .data import build_splits
            S = build_splits(DEFAULT_DATA, None if args.original_only else EXTRA_DATA, seed=args.seed)
            si = S["info"]
            pool = pd.concat([S["orig_train"], S["ext_train"]], ignore_index=True)
            tests = {"orig_test": S["orig_test"]}
            if len(S["ext_test"]):
                tests = {"ext_test": S["ext_test"], **tests}
            print(f"Training pool {len(pool)} rows; test sets: { {k: len(v) for k, v in tests.items()} }")
            result = train(pool, test_sets=tests, groups=pool["group"].to_numpy(),
                           cv_repeats=3 if args.original_only else 1, **opts)
            oi, ei = si["orig_data_info"], si["ext_data_info"]
            desc = [f"* Original sample `data/bugzilla_summary_sample.csv` (copy of `summaryList.csv`): "
                    f"{oi['rows_raw']} rows, {oi['rows_used']} after removing {oi['dropped_duplicates']} exact duplicate."]
            if ei:
                desc += [f"* Collected `data/bugzilla_core_verified_2006_2008.csv` (public Bugzilla REST API, see its "
                         f".provenance.json): {ei['rows_raw']} rows; dropped {ei['dropped_duplicates']} duplicates, "
                         f"{ei['dropped_conflicting_labels']} with conflicting labels, unknown labels "
                         f"{ei['dropped_unknown_label']}; {si.get('ext_removed_in_original_sample', 0)} already in the "
                         "original sample were excluded."]
            desc += [f"* Splits: {si['sizes']} (near-duplicate threshold {si['near_dup_threshold']} cosine, "
                     "group-aware). `orig_test` = the exact 20 rows held out by the previous version."]
    except BugSeverityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = result["metadata"]
    meta["data_description"] = desc
    (out.parent / "metrics.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    joblib.dump(result | {"metadata": meta}, out, compress=3)
    if not args.no_report:
        write_report(meta, desc)
    for name, m in meta["test_sets"].items():
        print(f"{name}: accuracy {m['accuracy']:.3f}, macro-F1 {m['macro']['f1']:.3f}, "
              f"Non-Critical recall {m['per_class']['Non-Critical']['recall']:.2f} (n={m['n_samples']})")
    print(f"Selected: {meta['algorithm']} | saved model to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
