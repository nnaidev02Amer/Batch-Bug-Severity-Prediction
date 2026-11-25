"""Shared fixtures. The dataset here is SYNTHETIC and exists only to test code paths."""
import joblib
import pandas as pd
import pytest

from src.preprocessing import load_training_data
from src.train import train

CRIT = ["crash segfault on startup null pointer", "data loss corrupts database file",
        "security vulnerability allows remote code execution", "browser hangs freeze deadlock",
        "assertion failure crash when loading page"]
NORM = ["button label misaligned in settings dialog", "wrong error message shown after login",
        "sorting order incorrect in table view", "tooltip shows stale value sometimes",
        "scrollbar jumps when resizing window"]
LOW = ["typo in documentation paragraph", "minor colour tweak requested for icon",
       "trivial spelling mistake in tooltip", "enhancement add optional toolbar shortcut",
       "cosmetic spacing difference in footer"]


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    rows = []
    for i in range(4):
        for t in CRIT:
            rows.append((f"{t} v{i}", "critical"))
        for t in NORM:
            rows.append((f"{t} v{i}", "normal"))
        for t in LOW:
            rows.append((f"{t} v{i}", "minor"))
    p = tmp_path_factory.mktemp("data") / "synthetic.csv"
    pd.DataFrame(rows, columns=["summary", "severity"]).to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def trained_bundle(synthetic_csv, tmp_path_factory):
    df, _ = load_training_data(synthetic_csv)
    result = train(df, seed=0, cv_repeats=1)
    path = tmp_path_factory.mktemp("model") / "m.joblib"
    joblib.dump(result, path)
    return path, result
