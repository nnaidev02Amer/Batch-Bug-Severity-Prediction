import joblib
import pandas as pd
import pytest

from src.preprocessing import load_training_data
from src.train import train
from src.utils import CLASSES, BugSeverityError


def test_training_produces_metrics_and_model(trained_bundle):
    _, result = trained_bundle
    meta = result["metadata"]
    assert meta["algorithm"] in meta["model_comparison"]
    assert len(meta["model_comparison"]) == 4
    tm = meta["test_metrics"]
    assert 0.0 <= tm["accuracy"] <= 1.0
    assert sum(sum(r) for r in tm["confusion_matrix"]) == tm["n_samples"]
    assert set(tm["per_class"]) == set(CLASSES)


def test_saved_model_reloads(trained_bundle):
    path, _ = trained_bundle
    bundle = joblib.load(path)
    assert {"pipeline", "metadata"} <= set(bundle)


def test_training_is_reproducible(synthetic_csv):
    df, _ = load_training_data(synthetic_csv)
    a = train(df, seed=1, cv_repeats=1)["metadata"]["test_metrics"]["accuracy"]
    b = train(df, seed=1, cv_repeats=1)["metadata"]["test_metrics"]["accuracy"]
    assert a == b


def test_training_rejects_missing_class():
    df = pd.DataFrame({"text": ["a crash"] * 10, "label": ["Critical"] * 10,
                       "raw_severity": ["critical"] * 10})
    with pytest.raises(BugSeverityError):
        train(df)
