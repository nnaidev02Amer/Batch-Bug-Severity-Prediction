"""Reproducible research experiments behind MODEL_REPORT.md.

Usage:  python -m src.experiments [--stages s0,s1,...] [--fresh]

Writes every number to reports/experiments.json. Stages are cached in that JSON;
--fresh recomputes. (Stage s7, error analysis, is implemented but has not been run yet.)

Rules followed:
* Model/feature/normaliser selection uses cross-validation on TRAINING data only.
* The two test sets (orig_test = previous 20 held-out rows; ext_test = held-out
  collected reports) are only used to evaluate configurations already chosen by CV,
  and for post-hoc diagnostics (learning curve, error analysis) that choose nothing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict, train_test_split

from .data import build_splits
from .evaluation import evaluate_predictions
from .modeling import (CLASSIFIERS, FEATURE_SETS, align_proba, calibrated, config_name,
                       cv_score, decide, make_cv_splits, make_pipeline, probability_metrics,
                       risk_coverage)
from .preprocessing import NORMALIZERS, combine_fields, normalize_label, read_csv
from .utils import CLASSES, DEFAULT_DATA, DEFAULT_MODEL, REPORTS_DIR

OUT_JSON = REPORTS_DIR / "experiments.json"
SEED = 42
OLD_CONFIG = ("Random Forest", "word+char", "technical")  # previous deployed choice


def _metrics(y, pred) -> dict:
    m = evaluate_predictions(list(y), list(pred))
    m.pop("report_text")
    return m


def _short(m: dict) -> dict:
    return {"n": m["n_samples"], "accuracy": m["accuracy"], "macro_f1": m["macro"]["f1"],
            "weighted_f1": m["weighted"]["f1"],
            "noncritical_recall": m["per_class"]["Non-Critical"]["recall"],
            "noncritical_precision": m["per_class"]["Non-Critical"]["precision"]}


def _grid(X, y, groups, n_splits, repeats, models=CLASSIFIERS, features=FEATURE_SETS,
          normalizers=tuple(NORMALIZERS)) -> dict:
    splits = make_cv_splits(y, groups, n_splits, repeats, SEED)
    res = {}
    for norm in normalizers:
        for feat in features:
            for model in models:
                t = time.time()
                r = cv_score(make_pipeline(model, feat, norm, SEED), X, y, splits)
                r["seconds"] = round(time.time() - t, 1)
                res[config_name(model, feat, norm)] = r
                print(f"  {config_name(model, feat, norm):58s} macroF1 {r['cv_macro_f1_mean']:.3f}"
                      f"±{r['cv_macro_f1_std']:.3f} acc {r['cv_accuracy_mean']:.3f} "
                      f"NC-rec {r['cv_noncritical_recall_mean']:.2f} ({r['seconds']}s)", flush=True)
    return {"cv": {"n_splits": len(splits) // repeats, "repeats": repeats, "group_aware": True,
                   "n_rows": int(len(y))}, "results": res}


def _best(grid: dict) -> tuple[str, str, str]:
    name = max(grid["results"], key=lambda k: grid["results"][k]["cv_macro_f1_mean"])
    return tuple(name.split(" | "))


def stage_s0(S):
    """Reproduce the previous result exactly (raw 99 rows, old split, old config)."""
    raw = read_csv(DEFAULT_DATA)
    X = np.array([combine_fields(r) for r in raw.to_dict("records")])
    y = np.array([normalize_label(s) for s in raw["severity"]])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pipe = make_pipeline(*OLD_CONFIG, seed=SEED).fit(Xtr, ytr)
    m = _metrics(yte, pipe.predict(Xte))
    return {"description": "Previous protocol re-run: raw 99 rows, seed-42 split (79/20), "
                           "Random Forest | word+char | technical",
            "test": _short(m), "confusion_matrix": m["confusion_matrix"]}


def _pool(S):
    tr = pd.concat([S["orig_train"], S["ext_train"]], ignore_index=True)
    return tr["text"].to_numpy(), tr["label"].to_numpy(), tr["group"].to_numpy(), tr


def stage_s1(S):
    """Feature/normaliser/model grid on the ORIGINAL training rows only."""
    d = S["orig_train"]
    return _grid(d["text"].to_numpy(), d["label"].to_numpy(), d["group"].to_numpy(), 5, 3)


def stage_s2(S):
    """Same grid on the expanded training pool (orig_train + ext_train)."""
    X, y, g, _ = _pool(S)
    return _grid(X, y, g, 5, 1)


def _eval_on_tests(pipe, S) -> dict:
    out = {}
    for k in ("orig_test", "ext_test"):
        m = _metrics(S[k]["label"], pipe.predict(S[k]["text"].to_numpy()))
        out[k] = _short(m) | {"confusion_matrix": m["confusion_matrix"],
                              "per_class": m["per_class"]}
    return out


def stage_s3(S, R):
    """Evaluate CV-selected configurations once on both test sets."""
    X, y, _, _ = _pool(S)
    o = S["orig_train"]
    rows = {}
    sel_orig, sel_pool = _best(R["s1"]), _best(R["s2"])
    plan = [
        ("A. previous config, trained on orig_train (78)", OLD_CONFIG, o["text"], o["label"]),
        ("B. best-by-CV on orig_train, trained on orig_train (78)", sel_orig, o["text"], o["label"]),
        (f"C. previous config, trained on expanded pool ({len(y)})", OLD_CONFIG, X, y),
        (f"D. best-by-CV on pool, trained on expanded pool ({len(y)})", sel_pool, X, y),
    ]
    for label, cfg, Xt, yt in plan:
        pipe = make_pipeline(*cfg, seed=SEED).fit(np.asarray(Xt), np.asarray(yt))
        rows[label] = {"config": config_name(*cfg)} | _eval_on_tests(pipe, S)
        print(f"  {label}: {rows[label]['config']}  ext macroF1 {rows[label]['ext_test']['macro_f1']:.3f}"
              f"  orig macroF1 {rows[label]['orig_test']['macro_f1']:.3f}", flush=True)
    # The artifact deployed by the previous version (fit on all 99 original rows,
    # including orig_test -> only its ext_test number is a fair measurement).
    if DEFAULT_MODEL.exists():
        old = joblib.load(DEFAULT_MODEL)
        if old["metadata"].get("n_total") == 99:
            e = S["ext_test"]
            m = _metrics(e["label"], old["pipeline"].predict(e["text"].to_numpy()))
            rows["Previous deployed artifact (fit on all 99 rows)"] = {
                "config": config_name(*OLD_CONFIG),
                "ext_test": _short(m) | {"confusion_matrix": m["confusion_matrix"],
                                         "per_class": m["per_class"]}}
    return {"selected_on_orig_cv": config_name(*sel_orig), "selected_on_pool_cv": config_name(*sel_pool),
            "rows": rows}


def stage_s4(S, R):
    """Learning curve of the pool-selected config (diagnostic only)."""
    cfg = _best(R["s2"])
    X, y, _, _ = _pool(S)
    e = S["ext_test"]
    pts = []
    for n in (78, 150, 300, 600, 1200, 2400, len(y)):
        if n < len(y):
            idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=SEED)
        else:
            idx = np.arange(len(y))
        pipe = make_pipeline(*cfg, seed=SEED).fit(X[idx], y[idx])
        m = _short(_metrics(e["label"], pipe.predict(e["text"].to_numpy())))
        pts.append({"n_train": int(n)} | m)
        print(f"  n={n:5d} ext macroF1 {m['macro_f1']:.3f} acc {m['accuracy']:.3f} NC-rec {m['noncritical_recall']:.2f}", flush=True)
    return {"config": config_name(*cfg), "points": pts}


def stage_s5(S, R):
    """Label-mapping evidence: out-of-fold predictions per RAW Bugzilla label,
    plus a sensitivity check of alternative mappings (CV on the pool only)."""
    cfg = _best(R["s2"])
    X, y, g, pool = _pool(S)
    splits = make_cv_splits(y, g, 5, 1, SEED)
    oof = cross_val_predict(make_pipeline(*cfg, seed=SEED), X, y, cv=splits)
    ct = pd.crosstab(pool["raw_severity"], pd.Series(oof, name="predicted")).reindex(columns=CLASSES, fill_value=0)
    per_raw = {raw: {c: int(ct.loc[raw, c]) for c in CLASSES} for raw in ct.index}

    alt = {}
    variants = {
        "current (major->Critical, enhancement->Non-Critical)": pool,
        "major->Normal": pool.assign(label=np.where(pool["raw_severity"] == "major", "Normal", pool["label"])),
        "enhancement excluded": pool[pool["raw_severity"] != "enhancement"],
        "major->Normal and enhancement excluded": pool.assign(
            label=np.where(pool["raw_severity"] == "major", "Normal", pool["label"]))
            .loc[lambda d: d["raw_severity"] != "enhancement"],
    }
    for name, d in variants.items():
        Xv, yv, gv = d["text"].to_numpy(), d["label"].to_numpy(), d["group"].to_numpy()
        r = cv_score(make_pipeline(*cfg, seed=SEED), Xv, yv, make_cv_splits(yv, gv, 5, 1, SEED))
        alt[name] = {"n": int(len(d)), "class_counts": pd.Series(yv).value_counts().to_dict()} | {
            k: r[k] for k in ("cv_accuracy_mean", "cv_macro_f1_mean", "cv_noncritical_recall_mean")}
        print(f"  {name:55s} macroF1 {r['cv_macro_f1_mean']:.3f} NC-rec {r['cv_noncritical_recall_mean']:.2f}", flush=True)
    return {"config": config_name(*cfg), "oof_by_raw_label": per_raw, "mapping_sensitivity": alt}


def stage_s6(S, R):
    """Calibration and decision rule, judged on out-of-fold predictions of the pool."""
    cfg = _best(R["s2"])
    X, y, g, _ = _pool(S)
    splits = make_cv_splits(y, g, 5, 1, SEED)
    priors = {c: float((y == c).mean()) for c in CLASSES}
    base = make_pipeline(*cfg, seed=SEED)
    variants = {}
    if hasattr(base, "predict_proba") and cfg[0] != "Linear SVM":
        variants["uncalibrated"] = base
    variants["sigmoid"] = calibrated(base, "sigmoid", SEED)
    variants["isotonic"] = calibrated(base, "isotonic", SEED)
    out = {"config": config_name(*cfg), "priors": priors, "variants": {}}
    for name, est in variants.items():
        t = time.time()
        proba = cross_val_predict(est, X, y, cv=splits, method="predict_proba")
        proba = align_proba(sorted(set(y)), proba)  # cross_val_predict columns = sorted labels
        v = {"probability": probability_metrics(y, proba)}
        for rule, pr in (("argmax", None), ("prior_corrected", priors)):
            lab = decide(proba, pr)
            v[rule] = _short(_metrics(y, lab)) | {"risk_coverage": risk_coverage(y, proba, lab)}
        out["variants"][name] = v
        print(f"  {name:12s} logloss {v['probability']['log_loss']:.3f} ECE {v['probability']['ece']:.3f} | "
              f"argmax F1 {v['argmax']['macro_f1']:.3f} NC {v['argmax']['noncritical_recall']:.2f} | "
              f"prior-corr F1 {v['prior_corrected']['macro_f1']:.3f} NC {v['prior_corrected']['noncritical_recall']:.2f}"
              f" ({time.time()-t:.0f}s)", flush=True)
    return out


def top_features(model, text: str, label: str, k: int = 6) -> list[str]:
    """Features of ``text`` that push the model most towards ``label``.
    Linear models: tf-idf x class weight; NB: tf-idf x (log p(f|c) - mean); RF:
    tf-idf x global importance (not class specific)."""
    pipe = model.calibrated_classifiers_[0].estimator if hasattr(model, "calibrated_classifiers_") else model
    feats, clf = pipe.named_steps["features"], pipe.named_steps["clf"]
    x = feats.transform([text]).toarray()[0]
    names = feats.get_feature_names_out()
    ci = list(clf.classes_).index(label)
    if hasattr(clf, "coef_"):
        w = clf.coef_[ci]
    elif hasattr(clf, "feature_log_prob_"):
        w = clf.feature_log_prob_[ci] - clf.feature_log_prob_.mean(axis=0)
    else:
        w = clf.feature_importances_
    contrib = x * w
    idx = [i for i in np.argsort(-contrib)[:k] if contrib[i] > 0]
    return [str(names[i]).replace("word__", "w:").replace("char__", "c:") for i in idx]


def stage_s7(S, R):
    """Error analysis of the deployed recipe (fit on the training pool)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from .preprocessing import clean_text, tokenize
    from .train import DEFAULT_CALIBRATION, DEFAULT_DECISION
    cfg = _best(R["s2"])  # same recipe train.py selects (identical CV splits)
    X, y, _, pool = _pool(S)
    priors = {c: float((y == c).mean()) for c in CLASSES} if DEFAULT_DECISION == "prior_corrected" else None
    model = calibrated(make_pipeline(*cfg, seed=SEED), DEFAULT_CALIBRATION, SEED).fit(X, y)
    f = model.calibrated_classifiers_[0].estimator.named_steps["features"]
    word_vec = f.transformer_list[0][1] if cfg[1] == "word+char" else (f if cfg[1] == "word" else None)
    vocab = set(word_vec.vocabulary_) if word_vec is not None else None
    nn_vec = TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb", ngram_range=(3, 5)).fit(X)
    Xn = nn_vec.transform(X)
    out = {"config": config_name(*cfg), "calibration": DEFAULT_CALIBRATION, "decision": DEFAULT_DECISION}
    for split in ("orig_test", "ext_test"):
        d = S[split]
        proba = align_proba(model, model.predict_proba(d["text"].to_numpy()))
        pred = decide(proba, priors)
        sims = (nn_vec.transform(d["text"]) @ Xn.T).toarray()
        errs = []
        for i in np.where(pred != d["label"].to_numpy())[0]:
            r = d.iloc[i]
            j = int(sims[i].argmax())
            toks = tokenize(clean_text(r["text"]))
            errs.append({
                "bug_id": int(r["bug_id"]), "actual": r["label"], "raw_label": r["raw_severity"],
                "predicted": str(pred[i]), "confidence": float(proba[i].max()),
                "probabilities": {c: round(float(p), 3) for c, p in zip(CLASSES, proba[i])},
                "summary": r["text"], "top_features_for_prediction": top_features(model, r["text"], pred[i]),
                "nearest_train": {"similarity": round(float(sims[i, j]), 3), "label": str(y[j]),
                                  "raw_label": str(pool.iloc[j]["raw_severity"]), "summary": str(X[j])},
                "known_word_tokens": None if vocab is None else int(sum(t in vocab for t in toks)),
                "signals": {
                    "mapping_related_raw_label": r["raw_severity"] in ("major", "enhancement"),
                    "minority_class": r["label"] == "Non-Critical",
                    "similar_train_report_with_predicted_label": bool(sims[i, j] >= 0.5 and y[j] == pred[i]),
                    "low_confidence": bool(proba[i].max() < 0.6),
                }})
        n = len(d)
        sig = pd.DataFrame([e["signals"] for e in errs]) if errs else pd.DataFrame()
        out[split] = {
            "n": n, "n_errors": len(errs),
            "errors_by_actual_predicted": pd.Series([f"{e['actual']} -> {e['predicted']}" for e in errs])
                                            .value_counts().to_dict(),
            "error_rate_by_raw_label": {raw: {"n": int((d["raw_severity"] == raw).sum()),
                                              "errors": int(sum(e["raw_label"] == raw for e in errs))}
                                        for raw in d["raw_severity"].value_counts().index},
            "signal_share": {k: float(v) for k, v in sig.mean().items()} if len(sig) else {},
            "accuracy_low_vs_high_confidence": {
                "low": float((pred == d["label"].to_numpy())[proba.max(1) < 0.6].mean()),
                "high": float((pred == d["label"].to_numpy())[proba.max(1) >= 0.6].mean())},
            "errors": errs if split == "orig_test" else sorted(errs, key=lambda e: -e["confidence"]),
        }
        print(f"  {split}: {len(errs)} errors / {n}; {out[split]['errors_by_actual_predicted']}", flush=True)
    return out


STAGES = {"s0": stage_s0, "s1": stage_s1, "s2": stage_s2, "s3": stage_s3, "s4": stage_s4,
          "s5": stage_s5, "s6": stage_s6, "s7": stage_s7}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args(argv)
    REPORTS_DIR.mkdir(exist_ok=True)
    R = {} if args.fresh or not OUT_JSON.exists() else json.loads(OUT_JSON.read_text(encoding="utf-8"))
    S = build_splits(seed=SEED)
    R["splits"] = S["info"]
    for st in args.stages.split(","):
        print(f"== {st}: {STAGES[st].__doc__.strip().splitlines()[0]}", flush=True)
        t = time.time()
        fn = STAGES[st]
        R[st] = fn(S) if fn.__code__.co_argcount == 1 else fn(S, R)
        R[st]["_seconds"] = round(time.time() - t, 1)
        OUT_JSON.write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
