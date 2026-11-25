"""UI tests for the redesigned dashboard (Streamlit AppTest, headless) and the pure
helpers in ui/content.py. These run the real app with the real trained model; they
do not drive a browser."""
import io

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.predict import SeverityPredictor
from src.utils import CLASSES, DEFAULT_MODEL, LOW_CONFIDENCE_MESSAGE, PRIORITY, ROOT, TEXT_FIELDS
from ui import content as C

APP = str(ROOT / "app.py")
needs_model = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="model not trained yet")
EXPERIMENTAL = "This prediction is experimental. The model was trained on a limited dataset."


# ---------------------------------------------------------------- helpers
def app_on(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def button(at, label):
    return next(b for b in at.button if b.label == label)


def field_values(at) -> dict:
    return {f: at.session_state[f"f_{f}"] for f in C.FORM_FIELDS}


def all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [w.value for w in at.warning] + [e.value for e in at.error] \
        + [i.value for i in at.info] + [c.value for c in at.caption] + [s.value for s in at.success]
    return "\n".join(map(str, parts))


# ---------------------------------------------------------------- pure content helpers
def test_examples_are_complete_and_unlabelled():
    assert list(C.EXAMPLES) == ["Login failure", "Application crash", "UI display issue", "Minor text issue"]
    for name, ex in C.EXAMPLES.items():
        assert set(ex) == set(C.FORM_FIELDS)
        assert ex["title"] and ex["description"]
        assert "severity" not in " ".join(ex).lower()
    assert "SAMPLE" in C.EXAMPLES["Application crash"]["error"]   # illustrative error is marked


def test_missing_required():
    assert C.missing_required({"title": "x", "description": "y"}) == []
    assert C.missing_required({"title": " ", "description": "y"}) == ["What is the problem?"]
    assert len(C.missing_required({})) == 2


def test_column_guide_matches_prediction_schema():
    assert [r["Column"] for r in C.COLUMN_GUIDE] == TEXT_FIELDS
    assert set(C.TEMPLATE_COLUMNS) <= set(TEXT_FIELDS)


def test_template_csv_is_valid_and_predictable(tmp_path):
    raw = C.to_csv_bytes(C.template_dataframe())
    assert raw.startswith(b"\xef\xbb\xbf")                       # BOM for Excel
    df = C.read_upload(raw)
    assert list(df.columns) == C.TEMPLATE_COLUMNS and len(df) == 3
    rep = C.inspect_upload(df)
    assert not rep["missing_required"] and rep["valid"] == 3 and rep["invalid"] == 0
    assert "severity" not in df.columns


def test_sample_csv_has_ten_demo_rows_without_labels():
    df = C.read_upload(C.to_csv_bytes(C.sample_reports_dataframe()))
    assert len(df) == 10 and df["report_id"].str.startswith("DEMO-").all()
    assert not any("severity" in c for c in df.columns)
    assert C.inspect_upload(df)["valid"] == 10


def test_inspect_upload_detects_problems():
    df = pd.DataFrame({"Title": ["Crash on save", "", "Crash on save", "the of"],
                       "Description": ["App closes", None, "App closes", ""], "ID": [1, 2, 3, 4]})
    rep = C.inspect_upload(df)
    assert rep["text_columns"] == ["title", "description"] and rep["other_columns"] == ["id"]
    assert rep["total"] == 4 and rep["valid"] == 2 and rep["invalid"] == 2
    assert rep["invalid_rows"] == [3, 5]                       # spreadsheet row numbers
    assert rep["duplicates"] == 1 and rep["duplicate_rows"] == [4]
    missing = C.inspect_upload(pd.DataFrame({"foo": ["x"], "bar": ["y"]}))
    assert missing["missing_required"] and missing["valid"] == 0


@pytest.mark.parametrize("data,msg", [
    (b"", "empty"), (b"   \n", "empty"), (b"PK\x03\x04xx", "Excel workbook"),
    (b"title\nfoo, bar\n", "more values than there are column headers")])
def test_invalid_uploads_raise_clear_errors(data, msg):
    from src.utils import BugSeverityError
    with pytest.raises(BugSeverityError, match=msg):
        C.read_upload(data)


def test_excel_encodings_accepted():
    for enc in ("utf-8-sig", "cp1252"):
        df = C.read_upload("title,description\nCafé page – broken,x\n".encode(enc))
        assert df.loc[0, "title"] == "Café page – broken"


@needs_model
def test_batch_prediction_preserves_original_columns_and_download():
    pred = SeverityPredictor()
    src = C.sample_reports_dataframe().rename(columns={"title": "Title", "report_id": "Report ID"})
    raw = C.to_csv_bytes(src)
    original = C.read_upload(raw)
    out = C.restore_original_headers(pred.predict_csv(io.BytesIO(raw)), original.columns)
    assert list(out.columns[: len(original.columns)]) == list(original.columns)   # names & order kept
    pd.testing.assert_frame_equal(out[original.columns], original)                # values untouched
    assert set(out["predicted_severity"]) <= set(CLASSES)
    assert (out["priority"] == out["predicted_severity"].map(PRIORITY)).all()
    back = pd.read_csv(io.BytesIO(C.results_csv(out)), encoding="utf-8-sig")
    assert list(back.columns) == list(out.columns) and len(back) == 10
    # predictions equal the predictor's own single predictions (nothing hard-coded)
    one = pred.predict_one(**{k: src.loc[0, k if k != "title" else "Title"] for k in C.FORM_FIELDS
                              if (k if k != "title" else "Title") in src})
    assert out.loc[0, "predicted_severity"] == one["severity"]


def test_results_csv_drops_confidence_only_when_unavailable():
    df = pd.DataFrame({"title": ["a"], "predicted_severity": ["Normal"], "priority": ["Medium Priority"],
                       "confidence": [float("nan")], "review": [LOW_CONFIDENCE_MESSAGE]})
    assert "confidence" not in pd.read_csv(io.BytesIO(C.results_csv(df)), encoding="utf-8-sig")
    df["confidence"] = 0.7
    assert "confidence" in pd.read_csv(io.BytesIO(C.results_csv(df)), encoding="utf-8-sig")


# ---------------------------------------------------------------- Predict page (AppTest)
@needs_model
def test_predict_form_renders_all_sections_and_fields():
    at = app_on("🔮 Predict")
    labels = [w.label for w in at.text_input] + [w.label for w in at.text_area]
    for f in C.FORM_FIELDS:
        assert any(lbl.startswith(C.FIELD_LABELS[f]) for lbl in labels), f
    assert at.text_input[0].placeholder == C.FIELD_PLACEHOLDERS["title"]
    text = all_text(at)
    assert "Basic bug information" in text and "Reproduction details" in text and "Technical details" in text
    assert "Try Example Bug Reports" in text and "Demonstration only" in text
    assert {"Load Example", "Clear Form"} <= {b.label for b in at.button}


@needs_model
@pytest.mark.parametrize("name", list(C.EXAMPLES))
def test_load_example_populates_every_field(name):
    at = app_on("🔮 Predict")
    at.selectbox(key="example_choice").set_value(name).run()
    button(at, "Load Example").click().run()
    assert not at.exception
    assert field_values(at) == C.EXAMPLES[name]
    assert at.text_input[0].value == C.EXAMPLES[name]["title"]


@needs_model
def test_clear_form_resets_fields_and_result():
    at = app_on("🔮 Predict")
    button(at, "Load Example").click().run()
    button(at, "🔮 Predict Severity").click().run()
    assert at.session_state["pred_result"] is not None
    button(at, "Clear Form").click().run()
    assert all(v == "" for v in field_values(at).values())
    assert at.session_state["pred_result"] is None


@needs_model
@pytest.mark.parametrize("title,desc,expected", [
    ("", "", "empty"),
    ("App crashes on save", "", "Describe the bug"),
    ("", "The app closes when I press save", "What is the problem?"),
    ("the of and", "a an the", "empty"),            # only stop-words: rejected by the predictor
])
def test_required_and_invalid_input_validation(title, desc, expected):
    at = app_on("🔮 Predict")
    at.text_input[0].set_value(title)
    at.text_area[0].set_value(desc)
    button(at, "🔮 Predict Severity").click().run()
    assert at.error and expected in at.error[0].value
    assert at.session_state["pred_result"] is None


@needs_model
def test_prediction_with_required_fields_only_matches_model():
    at = app_on("🔮 Predict")
    title, desc = "Browser crashes when opening a page", "Null pointer crash on startup"
    at.text_input[0].set_value(title)
    at.text_area[0].set_value(desc)
    button(at, "🔮 Predict Severity").click().run()
    assert not at.exception and not at.error
    expected = SeverityPredictor().predict_one(title=title, description=desc, steps="", expected="",
                                               actual="", error="")
    r = at.session_state["pred_result"]
    assert r["severity"] == expected["severity"] and r["priority"] == PRIORITY[r["severity"]]
    html = "\n".join(m.value for m in at.markdown)
    assert expected["severity"] in html and expected["priority"] in html
    text = all_text(at)
    assert EXPERIMENTAL in text and "Please review manually" in text
    if expected["confidence"] is not None:
        assert any("Model confidence" in m.label for m in at.metric)
        assert at.session_state["pred_result"]["confidence"] == pytest.approx(expected["confidence"])
        if expected["low_confidence"]:
            assert LOW_CONFIDENCE_MESSAGE in text


@needs_model
def test_prediction_with_all_optional_fields():
    at = app_on("🔮 Predict")
    at.selectbox(key="example_choice").set_value("Login failure").run()
    button(at, "Load Example").click().run()
    button(at, "🔮 Predict Severity").click().run()
    assert not at.exception and not at.error
    assert at.session_state["pred_result"]["severity"] in CLASSES


# ---------------------------------------------------------------- Batch page (AppTest)
@needs_model
def test_batch_page_renders_guide_and_downloads():
    at = app_on("📄 Batch Prediction")
    labels = [d.label for d in at.download_button]
    assert "Download CSV Template" in labels and "Download Sample Bug Reports" in labels
    text = all_text(at)
    assert "Upload multiple bug reports in a CSV file" in text
    assert len(at.file_uploader) == 1 and at.file_uploader[0].allowed_type == [".csv"]
    assert {e.label for e in at.expander} >= {"📘 How to Prepare Your CSV File", "❓ Frequently asked questions"}


def _upload(at, data: bytes, name="reports.csv"):
    at.file_uploader[0].set_value((name, data, "text/csv")).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@needs_model
def test_batch_upload_preview_predict_and_download():
    at = app_on("📄 Batch Prediction")
    _upload(at, C.to_csv_bytes(C.sample_reports_dataframe()))
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Records detected"] == "10" and metrics["Valid records"] == "10"
    assert metrics["Invalid records"] == "0" and metrics["Missing required columns"] == "None"
    assert len(at.dataframe) >= 2                                  # column guide + preview
    run = button(at, "🔮 Predict All Bug Severities")
    assert not run.disabled
    run.click().run()
    assert not at.exception
    key, out = at.session_state["batch_result"]
    assert len(out) == 10 and list(out.columns[:8]) == C.SAMPLE_COLUMNS
    expected = SeverityPredictor().predict_batch(C.sample_reports_dataframe())
    assert list(out["predicted_severity"]) == list(expected["predicted_severity"])
    metrics = {m.label: m.value for m in at.metric}
    for c in CLASSES:
        icon_label = next(lbl for lbl in metrics if lbl.endswith(c))
        assert metrics[icon_label] == str(int((expected["predicted_severity"] == c).sum()))
    assert "📥 Download Prediction Results" in [d.label for d in at.download_button]
    # filter: only Critical rows shown
    at.pills(key="batch_filter").set_value(["Critical"]).run()
    shown = at.dataframe[-1].value
    assert set(shown["Predicted Severity"]) <= {"Critical"}


@needs_model
@pytest.mark.parametrize("data,expect", [
    (b"foo,bar\n1,2\n", "No supported text column found"),
    (b"title,description\n", "no bug report rows"),
    (b"", "could not be read"),
    (b"PK\x03\x04binary", "Excel workbook"),
])
def test_batch_invalid_uploads_show_errors_and_block_prediction(data, expect):
    at = _upload(app_on("📄 Batch Prediction"), data)
    assert at.error and expect in " ".join(e.value for e in at.error)
    runs = [b for b in at.button if b.label == "🔮 Predict All Bug Severities"]
    assert not runs or runs[0].disabled


@needs_model
def test_batch_empty_rows_and_duplicates_warned():
    data = b"title,description\nCrash on save,App closes\n,\nCrash on save,App closes\n"
    at = _upload(app_on("📄 Batch Prediction"), data)
    text = " ".join(w.value for w in at.warning)
    assert "no usable text" in text and "repeat an earlier report" in text
    button(at, "🔮 Predict All Bug Severities").click().run()
    _, out = at.session_state["batch_result"]
    assert out.loc[1, "predicted_severity"].startswith("Invalid")


# ---------------------------------------------------------------- navigation
@needs_model
def test_home_quick_navigation_buttons():
    at = AppTest.from_file(APP, default_timeout=90).run()
    for label, page in [("Open Predict", "🔮 Predict"), ("Open Batch Prediction", "📄 Batch Prediction"),
                        ("Open Analytics", "📊 Analytics")]:
        at.sidebar.radio[0].set_value("🏠 Home").run()
        button(at, label).click().run()
        assert not at.exception and at.sidebar.radio[0].value == page
