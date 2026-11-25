import io

import joblib
import pandas as pd
import pytest

from src.predict import SeverityPredictor
from src.utils import CLASSES, PRIORITY, BugSeverityError, ModelNotFoundError


@pytest.fixture(scope="module")
def predictor(trained_bundle):
    return SeverityPredictor(trained_bundle[0])


def test_missing_model_file(tmp_path):
    with pytest.raises(ModelNotFoundError):
        SeverityPredictor(tmp_path / "none.joblib")


def test_corrupt_model_file(tmp_path):
    p = tmp_path / "bad.joblib"
    joblib.dump({"x": 1}, p)
    with pytest.raises(BugSeverityError):
        SeverityPredictor(p)


def test_single_prediction_shape_and_priority(predictor):
    r = predictor.predict_one(title="App crash segfault on startup", description="null pointer crash")
    assert r["severity"] in CLASSES
    assert r["priority"] == PRIORITY[r["severity"]]
    if r["probabilities"]:
        assert abs(sum(r["probabilities"].values()) - 1) < 1e-6


def test_prediction_comes_from_model_not_constants(predictor):
    a = predictor.predict_one(title="crash segfault data loss corrupts database")["severity"]
    b = predictor.predict_one(title="typo spelling mistake documentation cosmetic")["severity"]
    assert a == "Critical" and b == "Non-Critical"


def test_empty_prediction_raises(predictor):
    with pytest.raises(BugSeverityError):
        predictor.predict_one(title="", description="  ")
    with pytest.raises(BugSeverityError):
        predictor.predict_one()


def test_batch_prediction(predictor):
    df = pd.DataFrame({"title": ["crash on startup segfault", "", "typo in docs"], "id": [1, 2, 3]})
    out = predictor.predict_batch(df)
    assert out.loc[1, "predicted_severity"].startswith("Invalid")
    assert out.loc[0, "predicted_severity"] in CLASSES
    assert len(out) == 3 and "priority" in out


def test_batch_missing_text_column(predictor):
    with pytest.raises(BugSeverityError):
        predictor.predict_batch(pd.DataFrame({"id": [1, 2]}))


def test_batch_empty_frame(predictor):
    with pytest.raises(BugSeverityError):
        predictor.predict_batch(pd.DataFrame({"title": []}))


def test_batch_csv_roundtrip(predictor):
    out = predictor.predict_csv(io.StringIO("summary,bug_id\ncrash segfault,1\nspelling typo,2\n"))
    reparsed = pd.read_csv(io.StringIO(out.to_csv(index=False)))
    assert len(reparsed) == 2 and "predicted_severity" in reparsed
