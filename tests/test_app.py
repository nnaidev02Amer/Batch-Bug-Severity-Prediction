"""Headless smoke tests of the Streamlit UI (requires models/trained_model.joblib)."""
import pytest
from streamlit.testing.v1 import AppTest

from src.utils import DEFAULT_MODEL, ROOT

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="model not trained yet")

APP = str(ROOT / "app.py")
PAGES = ["🏠 Home", "🔮 Predict", "📄 Batch Prediction", "📊 Analytics", "🧠 Model Info", "ℹ️ About"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(page):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_predict_page_single_prediction_and_empty_error():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("🔮 Predict").run()
    at.button[0].click().run()  # empty form
    assert at.error and "empty" in at.error[0].value.lower()
    at.text_input[0].set_value("Crash with assertion failure on startup")
    at.text_area[0].set_value("Browser crashes with null pointer dereference when opening a page")
    at.button[0].click().run()
    assert not at.exception
    assert any("Please review manually" in w.value for w in at.warning)
