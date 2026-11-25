"""Building blocks shared by training (src/train.py) and research (src/experiments.py):
feature extractors, classifiers, group-aware CV, calibration metrics."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .preprocessing import NORMALIZERS, STOPWORDS, TOKEN_PATTERN
from .utils import CLASSES

FEATURE_SETS = ("word", "char", "word+char")
CLASSIFIERS = ("Logistic Regression", "Linear SVM", "Multinomial Naive Bayes", "Random Forest")

FEATURE_DESCRIPTIONS = {
    "word": "TF-IDF word 1-2-grams",
    "char": "TF-IDF char_wb 3-5-grams",
    "word+char": "TF-IDF word 1-2-grams + char_wb 3-5-grams",
}


def make_features(kind: str = "word+char", normalizer: str = "technical"):
    """TF-IDF feature extractor. ``normalizer`` is a key of NORMALIZERS."""
    pre = NORMALIZERS[normalizer]
    word = TfidfVectorizer(preprocessor=pre, token_pattern=TOKEN_PATTERN,
                           stop_words=sorted(STOPWORDS), ngram_range=(1, 2),
                           sublinear_tf=True, min_df=1)
    char = TfidfVectorizer(preprocessor=pre, analyzer="char_wb",
                           ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    if kind == "word":
        return word
    if kind == "char":
        return char
    if kind == "word+char":
        return FeatureUnion([("word", word), ("char", char)])
    raise ValueError(f"Unknown feature set {kind!r}; choose from {FEATURE_SETS}")


def make_classifier(name: str, seed: int = 42):
    """The four compared classifiers. Imbalance: class_weight='balanced'
    (LR/SVM/RF) or uniform class priors (NB)."""
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0, random_state=seed)
    if name == "Linear SVM":
        return LinearSVC(class_weight="balanced", C=0.5, random_state=seed)
    if name == "Multinomial Naive Bayes":
        return MultinomialNB(alpha=0.3, fit_prior=False)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=seed, n_jobs=1)
    raise ValueError(f"Unknown classifier {name!r}")


def make_pipeline(model: str, features: str = "word+char", normalizer: str = "technical",
                  seed: int = 42) -> Pipeline:
    return Pipeline([("features", make_features(features, normalizer)),
                     ("clf", make_classifier(model, seed))])


def config_name(model: str, features: str, normalizer: str) -> str:
    return f"{model} | {features} | {normalizer}"


# --- cross-validation ----------------------------------------------------
def make_cv_splits(y, groups=None, n_splits: int = 5, repeats: int = 1, seed: int = 42):
    """Repeated stratified (group-aware if groups given) k-fold index splits.
    Group-aware splitting keeps near-duplicate reports in the same fold."""
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(n_splits, int(counts.min())))
    splits = []
    for r in range(repeats):
        if groups is None:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
            splits += list(cv.split(np.zeros(len(y)), y))
        else:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
            splits += list(cv.split(np.zeros(len(y)), y, groups))
    return splits


SCORING = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
    "noncritical_recall": make_scorer(recall_score, labels=["Non-Critical"], average="macro",
                                      zero_division=0),
    "noncritical_f1": make_scorer(f1_score, labels=["Non-Critical"], average="macro",
                                  zero_division=0),
}


def cv_score(pipe, X, y, splits) -> dict:
    """Mean/std of every SCORING metric over the given splits (vectoriser fitted per fold)."""
    res = cross_validate(clone(pipe), X, y, cv=splits, scoring=SCORING, n_jobs=1)
    out = {}
    for k in SCORING:
        v = res[f"test_{k}"]
        out[f"cv_{k}_mean"] = float(v.mean())
        out[f"cv_{k}_std"] = float(v.std())
    return out


# --- probabilities & calibration ----------------------------------------
def calibrated(pipe, method: str = "sigmoid", seed: int = 42, cv: int = 5):
    """Wrap a pipeline so it outputs calibrated probabilities. ensemble=False keeps
    ONE base model (lightweight) plus calibrators fitted on out-of-fold scores."""
    return CalibratedClassifierCV(clone(pipe), method=method, ensemble=False,
                                  cv=StratifiedKFold(cv, shuffle=True, random_state=seed))


def align_proba(model_or_classes, proba) -> np.ndarray:
    """Reorder probability columns into CLASSES order (missing classes -> 0).
    Accepts a fitted model (uses ``classes_``) or the list of column labels."""
    classes = list(getattr(model_or_classes, "classes_", model_or_classes))
    return np.column_stack([proba[:, classes.index(c)] if c in classes else np.zeros(len(proba))
                            for c in CLASSES])


def decide(proba: np.ndarray, priors: dict | None = None) -> np.ndarray:
    """Turn CLASSES-ordered probabilities into labels. With ``priors`` the
    decision is prior-corrected (argmax p(c)/prior(c)), i.e. the balanced-error
    decision rule, so minority classes are not swamped by calibration."""
    p = np.asarray(proba, dtype=float)
    if priors:
        p = p / np.array([max(priors[c], 1e-9) for c in CLASSES])
    return np.array(CLASSES)[p.argmax(axis=1)]


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Top-label ECE: |accuracy - confidence| averaged over confidence bins."""
    proba = np.asarray(proba)
    conf = proba.max(axis=1)
    pred = np.array(CLASSES)[proba.argmax(axis=1)]
    correct = (pred == np.asarray(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def probability_metrics(y_true, proba) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    onehot = np.column_stack([(y_true == c).astype(float) for c in CLASSES])
    # sklearn's log_loss expects columns in sorted-label order; ours are in CLASSES order.
    order = sorted(CLASSES)
    sorted_proba = proba[:, [CLASSES.index(c) for c in order]]
    return {
        "log_loss": float(log_loss(y_true, np.clip(sorted_proba, 1e-9, 1), labels=order)),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
        "ece": expected_calibration_error(y_true, proba),
        "mean_confidence": float(np.max(proba, axis=1).mean()),
    }


def risk_coverage(y_true, proba, labels, thresholds=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> list[dict]:
    """For each confidence threshold: share of reports the model answers
    (coverage) and accuracy on those vs on the ones flagged for review."""
    y_true, labels = np.asarray(y_true), np.asarray(labels)
    conf = np.asarray(proba)[np.arange(len(labels)), [CLASSES.index(l) for l in labels]]
    rows = []
    for t in thresholds:
        keep = conf >= t
        rows.append({
            "threshold": t,
            "coverage": float(keep.mean()),
            "accuracy_confident": float((labels[keep] == y_true[keep]).mean()) if keep.any() else None,
            "accuracy_flagged": float((labels[~keep] == y_true[~keep]).mean()) if (~keep).any() else None,
            "n_flagged": int((~keep).sum()),
        })
    return rows
