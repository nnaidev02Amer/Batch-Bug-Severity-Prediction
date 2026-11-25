"""Shared constants and helpers for the Bug Severity Prediction System."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "bugzilla_summary_sample.csv"
# Additional real reports collected by scripts/collect_bugzilla.py (see data/README.md)
EXTRA_DATA = ROOT / "data" / "bugzilla_core_verified_2006_2008.csv"
REPORTS_DIR = ROOT / "reports"
LOW_CONFIDENCE_MESSAGE = "Low confidence – Manual review recommended."
DEFAULT_MODEL = ROOT / "models" / "trained_model.joblib"
REPORT_PATH = ROOT / "MODEL_REPORT.md"

CLASSES = ["Critical", "Normal", "Non-Critical"]

PRIORITY = {
    "Critical": "High Priority",
    "Normal": "Medium Priority",
    "Non-Critical": "Low Priority",
}

# Explicit, documented mapping of raw (Bugzilla-style) severity labels to the
# three project classes. Keys are compared after lower-casing and replacing
# "_" / " " with "-".
LABEL_MAP = {
    # Critical
    "critical": "Critical",
    "severe": "Critical",
    "blocker": "Critical",
    "major": "Critical",
    # Normal
    "normal": "Normal",
    "medium": "Normal",
    # Non-Critical
    "non-critical": "Non-Critical",
    "noncritical": "Non-Critical",
    "non-severe": "Non-Critical",
    "minor": "Non-Critical",
    "trivial": "Non-Critical",
    "enhancement": "Non-Critical",
    "low": "Non-Critical",
}

# Optional text columns that can be combined into one report text.
TEXT_FIELDS = ["title", "summary", "description", "steps", "expected", "actual", "error"]


class BugSeverityError(ValueError):
    """Raised for invalid user input (bad CSV, empty text, unknown labels...)."""


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file does not exist."""
