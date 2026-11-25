"""Tests for the reliability improvements: normalisers, near-duplicates, unknown
labels, class imbalance, calibration, model loading, prediction consistency and the
leakage-safe splits. Synthetic data here only exercises code paths."""
import importlib.util
import math

import joblib
import numpy as np
import pandas as pd
import pytest

from src.modeling import (align_proba, calibrated, decide, expected_calibration_error,
                          make_cv_splits, make_pipeline, probability_metrics, risk_coverage)
from src.predict import DEFAULT_CONFIDENCE_THRESHOLD, SeverityPredictor
from src.preprocessing import (NORMALIZERS, clean_text, clean_text_basic, clean_text_split,
                               expand_identifiers, load_training_data, near_duplicate_groups,
                               normalize_label, split_identifier)
from src.utils import CLASSES, LABEL_MAP, LOW_CONFIDENCE_MESSAGE, BugSeverityError


# ---------------------------------------------------------------- preprocessing
def test_split_identifier_camel_snake_and_scope():
    assert split_identifier("nsThebesImage::AllowedImageSize") == \
        ["ns", "Thebes", "Image", "Allowed", "Image", "Size"]
    assert split_identifier("JS_Assert") == ["JS", "Assert"]
    assert split_identifier("XMLHttpRequest") == ["XML", "Http", "Request"]


def test_expand_identifiers_keeps_original_tokens_and_marks_crash_signature():
    t = clean_text_split("Crash [@ nsThebesImage::AllowedImageSize] JS_Assert 0x1012279c")
    toks = t.split()
    assert "nsthebesimage::allowedimagesize" in toks and "js_assert" in toks  # originals kept
    assert {"thebes", "image", "allowed", "size", "assert"} <= set(toks)    # sub-words added
    assert "crashsignaturetoken" in toks and "0x1012279c" in toks
    assert expand_identifiers("plain words only") == "plain words only"
    assert expand_identifiers(None) == "" and clean_text_split(float("nan")) == ""


def test_basic_normalizer_strips_punctuation_unlike_technical():
    s = "Assertion at jsobj.c:3697 0x1012279c"
    assert "jsobj.c:3697" in clean_text(s)
    assert "jsobj.c:3697" not in clean_text_basic(s) and "3697" in clean_text_basic(s)
    assert set(NORMALIZERS) == {"basic", "technical", "technical+split"}


def test_near_duplicate_groups():
    texts = ['"Assertion failure: !SPROP_HAS_STUB_SETTER(sprop)"',
             "Assertion failure: !SPROP_HAS_STUB_SETTER(sprop)",
             "Typo in the preferences dialog title",
             "Crash when printing a large table"]
    g = near_duplicate_groups(texts, 0.8)
    assert g[0] == g[1]
    assert len({g[1], g[2], g[3]}) == 3
    assert list(near_duplicate_groups(["only one"])) == [0]


def test_duplicates_judged_on_cleaned_text_and_conflicts_dropped(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"summary": ["Crash On Start", "crash on   start", "button ugly", "button ugly",
                              "page hangs forever"],
                  "severity": ["critical", "critical", "minor", "normal", "critical"]}).to_csv(p, index=False)
    df, info = load_training_data(p)
    assert info["dropped_duplicates"] == 1
    assert info["dropped_conflicting_labels"] == 2   # both "button ugly" rows
    assert sorted(df["text"]) == ["Crash On Start", "page hangs forever"]


# ---------------------------------------------------------------- labels
@pytest.mark.parametrize("raw", ["S1", "S2", "S3", "S4", "--", "P1", "highest", "crash"])
def test_unknown_severity_labels_raise(raw):
    with pytest.raises(BugSeverityError, match="Unknown severity label"):
        normalize_label(raw)


def test_unknown_labels_are_dropped_and_counted_not_guessed(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"summary": ["a crash", "b glitch", "c typo", "d thing", "e thing"],
                  "severity": ["critical", "normal", "trivial", "S3", "S4"]}).to_csv(p, index=False)
    df, info = load_training_data(p)
    assert info["dropped_unknown_label"] == {"S3": 1, "S4": 1}
    assert len(df) == 3


def test_every_original_bugzilla_label_is_mapped_as_documented():
    expected = {"blocker": "Critical", "critical": "Critical", "major": "Critical",
                "normal": "Normal", "minor": "Non-Critical", "trivial": "Non-Critical",
                "enhancement": "Non-Critical"}
    assert {k: normalize_label(k) for k in expected} == expected
    assert set(LABEL_MAP.values()) == set(CLASSES)


def test_collector_label_rule_restores_enhancement():
    spec = importlib.util.spec_from_file_location("collect", "scripts/collect_bugzilla.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.original_label("normal", "enhancement") == ("enhancement", "type=enhancement")
    assert mod.original_label("normal", "defect") == ("normal", "severity")
    assert mod.original_label("critical", "enhancement") == ("critical", "severity")
    assert mod.original_label("S3", "defect") == ("S3", "severity")  # kept raw; loader drops it


# ---------------------------------------------------------------- class imbalance
def test_prior_corrected_decision_favours_minority_when_ambiguous():
    proba = np.array([[0.30, 0.45, 0.25]])          # Critical, Normal, Non-Critical
    priors = {"Critical": 0.45, "Normal": 0.48, "Non-Critical": 0.07}
    assert decide(proba)[0] == "Normal"
    assert decide(proba, priors)[0] == "Non-Critical"
    # a clear prediction is not overturned
    assert decide(np.array([[0.9, 0.08, 0.02]]), priors)[0] == "Critical"


def test_balanced_classifiers_and_stratified_folds_keep_minority(synthetic_csv):
    df, _ = load_training_data(synthetic_csv)
    df = pd.concat([df[df.label != "Non-Critical"], df[df.label == "Non-Critical"].head(5)])
    y = df["label"].to_numpy()
    for tr, te in make_cv_splits(y, None, 5, 1, 0):
        assert "Non-Critical" in set(y[tr]) and "Non-Critical" in set(y[te])
    for m in ("Logistic Regression", "Linear SVM", "Random Forest"):
        assert make_pipeline(m).named_steps["clf"].class_weight == "balanced"
    assert make_pipeline("Multinomial Naive Bayes").named_steps["clf"].fit_prior is False


def test_group_aware_splits_never_separate_a_group():
    y = np.array(["Critical", "Normal", "Non-Critical"] * 10)
    groups = np.repeat(np.arange(15), 2)
    for tr, te in make_cv_splits(y, groups, 3, 2, 0):
        assert not set(groups[tr]) & set(groups[te])


# ---------------------------------------------------------------- calibration
def test_probability_metrics_respect_column_order():
    # Regression test: sklearn log_loss assumes sorted label order; ours is CLASSES order.
    y = ["Critical", "Normal", "Non-Critical"]
    perfect = np.eye(3)
    m = probability_metrics(y, perfect)
    assert m["log_loss"] < 1e-6 and m["brier"] < 1e-9 and m["ece"] < 1e-9
    wrong = probability_metrics(y, perfect[:, [1, 2, 0]])
    assert wrong["log_loss"] > 5


def test_expected_calibration_error_detects_overconfidence():
    y = np.array(["Critical"] * 50 + ["Normal"] * 50)
    proba = np.tile([0.95, 0.04, 0.01], (100, 1))      # 95 % confident, 50 % right
    assert math.isclose(expected_calibration_error(y, proba), 0.45, abs_tol=1e-9)


def test_calibrated_model_outputs_valid_probabilities(synthetic_csv):
    df, _ = load_training_data(synthetic_csv)
    X, y = df["text"].to_numpy(), df["label"].to_numpy()
    model = calibrated(make_pipeline("Linear SVM"), "sigmoid", 0).fit(X, y)   # SVM gains predict_proba
    p = align_proba(model, model.predict_proba(X))
    assert p.shape == (len(X), 3)
    assert np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(1), 1)
    assert len(model.calibrated_classifiers_) == 1   # ensemble=False keeps one base model


def test_risk_coverage_is_monotonic():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet([1, 1, 1], size=200)
    y = np.array(CLASSES)[rng.integers(0, 3, 200)]
    rows = risk_coverage(y, proba, decide(proba))
    cov = [r["coverage"] for r in rows]
    assert cov == sorted(cov, reverse=True)
    assert all(r["n_flagged"] == round((1 - r["coverage"]) * 200) for r in rows)


# ---------------------------------------------------------------- model loading
@pytest.fixture(scope="module")
def predictor(trained_bundle):
    return SeverityPredictor(trained_bundle[0])


def test_trained_bundle_records_calibration_and_threshold(trained_bundle):
    meta = trained_bundle[1]["metadata"]
    assert meta["calibration"] == "sigmoid" and 0 < meta["confidence_threshold"] < 1
    assert meta["selection_metric"] == "CV macro-F1"
    assert {"probability", "risk_coverage"} <= set(meta["test_metrics"])


def test_legacy_bundle_without_new_metadata_still_loads(tmp_path, synthetic_csv):
    df, _ = load_training_data(synthetic_csv)
    pipe = make_pipeline("Random Forest").fit(df["text"], df["label"])
    p = tmp_path / "legacy.joblib"
    joblib.dump({"pipeline": pipe, "metadata": {"algorithm": "Random Forest"}}, p)
    pred = SeverityPredictor(p)
    assert pred.threshold == DEFAULT_CONFIDENCE_THRESHOLD and pred.priors is None
    assert pred.predict_one(title="crash segfault")["severity"] in CLASSES


def test_legacy_svm_bundle_without_probabilities_flags_review(tmp_path, synthetic_csv):
    df, _ = load_training_data(synthetic_csv)
    pipe = make_pipeline("Linear SVM").fit(df["text"], df["label"])
    p = tmp_path / "svm.joblib"
    joblib.dump({"pipeline": pipe, "metadata": {}}, p)
    r = SeverityPredictor(p).predict_one(title="crash segfault")
    assert r["confidence"] is None and r["low_confidence"] is None
    assert r["review_note"] == LOW_CONFIDENCE_MESSAGE


def test_bundle_with_non_model_object_rejected(tmp_path):
    p = tmp_path / "bad.joblib"
    joblib.dump({"pipeline": "not a model", "metadata": {}}, p)
    with pytest.raises(BugSeverityError):
        SeverityPredictor(p)


# ---------------------------------------------------------------- prediction consistency
def test_prediction_is_deterministic_and_reload_identical(predictor, trained_bundle):
    a = predictor.predict_one(title="browser crash null pointer", description="segfault on load")
    b = predictor.predict_one(title="browser crash null pointer", description="segfault on load")
    c = SeverityPredictor(trained_bundle[0]).predict_one(
        title="browser crash null pointer", description="segfault on load")
    assert a == b == c


def test_single_and_batch_predictions_agree(predictor):
    texts = ["crash segfault on startup", "typo in documentation", "tooltip misaligned in dialog"]
    batch = predictor.predict_batch(pd.DataFrame({"title": texts}))
    for i, t in enumerate(texts):
        one = predictor.predict_one(title=t)
        assert batch.loc[i, "predicted_severity"] == one["severity"]
        assert math.isclose(batch.loc[i, "confidence"], one["confidence"], rel_tol=1e-9)
        assert (batch.loc[i, "review"] == LOW_CONFIDENCE_MESSAGE) == one["low_confidence"]


def test_label_consistent_with_probabilities(predictor):
    r = predictor.predict_one(title="data loss corrupts database crash")
    p = np.array([[r["probabilities"][c] for c in CLASSES]])
    assert decide(p, predictor.priors)[0] == r["severity"]
    assert r["confidence"] == r["probabilities"][r["severity"]]


def test_low_confidence_message(predictor):
    old = predictor.threshold
    try:
        predictor.threshold = 1.01   # nothing can be that confident
        r = predictor.predict_one(title="crash segfault")
        assert r["low_confidence"] is True and r["review_note"] == LOW_CONFIDENCE_MESSAGE
        predictor.threshold = 0.0
        r = predictor.predict_one(title="crash segfault")
        assert r["low_confidence"] is False and r["review_note"] == ""
    finally:
        predictor.threshold = old
    assert LOW_CONFIDENCE_MESSAGE == "Low confidence – Manual review recommended."


# ---------------------------------------------------------------- real-data splits
def test_fixed_original_test_ids_and_no_leakage():
    from src.data import build_splits, original_test_ids
    ids = original_test_ids()
    assert len(ids) == 20
    S = build_splits()
    assert set(S["orig_test"]["bug_id"]) == ids
    train_ids = set(S["orig_train"]["bug_id"]) | set(S["ext_train"]["bug_id"])
    test_ids = set(S["orig_test"]["bug_id"]) | set(S["ext_test"]["bug_id"])
    assert not train_ids & test_ids
    train_groups = set(S["orig_train"]["group"]) | set(S["ext_train"]["group"])
    test_groups = set(S["orig_test"]["group"]) | set(S["ext_test"]["group"])
    assert not train_groups & test_groups          # no near-duplicate straddles train/test
    assert not set(S["ext_train"]["bug_id"]) & set(S["orig_train"]["bug_id"])
