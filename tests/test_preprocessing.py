import pandas as pd
import pytest

from src.preprocessing import (clean_text, combine_fields, load_training_data,
                               normalize_label, read_csv, tokenize, validate_report_text)
from src.utils import BugSeverityError


def test_label_mapping_case_and_spelling():
    assert normalize_label("CRITICAL") == "Critical"
    assert normalize_label(" Blocker ") == "Critical"
    assert normalize_label("non_critical") == "Non-Critical"
    assert normalize_label("Enhancement") == "Non-Critical"
    assert normalize_label("normal") == "Normal"


@pytest.mark.parametrize("bad", ["urgent!!", "", None, float("nan")])
def test_unknown_or_missing_label_raises(bad):
    with pytest.raises(BugSeverityError):
        normalize_label(bad)


def test_clean_text_keeps_technical_tokens_and_replaces_urls():
    t = clean_text("JS_Assert failed at 0x1012279c see https://x.org/a in nsDocShell.cpp:42")
    assert "js_assert" in t and "0x1012279c" in t and "nsdocshell.cpp:42" in t
    assert "urltoken" in t and "https" not in t


def test_tokenize_drops_stopwords_but_keeps_negation():
    toks = tokenize(clean_text("the app does not start"))
    assert "not" in toks and "the" not in toks


@pytest.mark.parametrize("bad", ["", "   ", "the of and", "!!! ???"])
def test_empty_report_rejected(bad):
    with pytest.raises(BugSeverityError):
        validate_report_text(bad)


def test_combine_fields_skips_missing():
    assert combine_fields({"title": "A", "description": None, "error": "boom"}) == "A . boom"


def test_load_training_data_drops_bad_rows(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"summary": ["crash on start", "crash on start", "ui glitch", "odd label here", "   "],
                  "severity": ["critical", "critical", "normal", "weird", "minor"]}).to_csv(p, index=False)
    df, info = load_training_data(p)
    assert len(df) == 2
    assert info["dropped_duplicates"] == 1
    assert info["dropped_empty_text"] == 1
    assert info["dropped_unknown_label"] == {"weird": 1}


def test_load_training_data_missing_columns(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(p, index=False)
    with pytest.raises(BugSeverityError):
        load_training_data(p)


def test_invalid_and_missing_csv(tmp_path):
    with pytest.raises(BugSeverityError):
        read_csv(tmp_path / "nope.csv")
    empty = tmp_path / "e.csv"
    empty.write_text("")
    with pytest.raises(BugSeverityError):
        read_csv(empty)


def test_original_single_quote_csv_loads():
    df, info = load_training_data("data/bugzilla_summary_sample.csv")
    assert info["rows_used"] > 90
    assert set(df.label) == {"Critical", "Normal", "Non-Critical"}
