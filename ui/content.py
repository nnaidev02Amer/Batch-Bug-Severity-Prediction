"""Static content and pure helpers for the dashboard (no Streamlit import, unit-testable).

Everything here is DEMONSTRATION material: the example bug reports and sample CSV
rows are illustrative inputs written for this app. They carry no severity labels,
are not Mozilla Bugzilla data and are never used for training or evaluation.
"""
from __future__ import annotations

import io

import pandas as pd

from src.preprocessing import combine_fields, validate_report_text
from src.utils import TEXT_FIELDS, BugSeverityError

# ---------------------------------------------------------------- single-report form
# Form field key -> keyword accepted by SeverityPredictor.predict_one
FORM_FIELDS = ["title", "description", "steps", "expected", "actual", "error"]
REQUIRED_FORM_FIELDS = ["title", "description"]

FIELD_LABELS = {
    "title": "What is the problem?",
    "description": "Describe the bug",
    "steps": "Steps to reproduce",
    "expected": "Expected behavior",
    "actual": "Actual behavior",
    "error": "Error message / stack trace",
}
FIELD_PLACEHOLDERS = {
    "title": "Example: Users cannot log in after clicking the Login button",
    "description": ("When users enter valid login credentials and click the Login button, the application "
                    "displays a server error instead of opening the dashboard."),
    "steps": ("1. Open the application.\n2. Navigate to the Login page.\n3. Enter valid credentials.\n"
              "4. Click Login.\n5. Observe the error."),
    "expected": "Example: The user should successfully log in and navigate to the dashboard.",
    "actual": "Example: The application displays HTTP 500 Internal Server Error.",
    "error": "HTTPError: 500 Internal Server Error\nat /api/auth/login",
}
FIELD_HELP = {
    "title": "Enter a short title describing the issue.",
    "description": "Explain what happened, when it happened and which functionality is affected.",
    "steps": "Explain the steps needed to reproduce the problem.",
    "expected": "What should happen when the functionality works correctly?",
    "actual": "What actually happens instead?",
    "error": "Paste the actual error from your browser console, terminal or application logs.",
}

# Illustrative inputs only - deliberately NO severity label attached.
EXAMPLES: dict[str, dict[str, str]] = {
    "Login failure": {
        "title": "Users cannot log in after clicking the Login button.",
        "description": ("When users enter valid email and password credentials and click Login, the application "
                        "displays HTTP 500 Internal Server Error instead of opening the dashboard."),
        "steps": "1. Open the application.\n2. Navigate to Login.\n3. Enter valid credentials.\n4. Click Login.",
        "expected": "The user should successfully log in.",
        "actual": "The application displays a server error.",
        "error": "HTTP 500 Internal Server Error.",
    },
    "Application crash": {
        "title": "Application crashes when uploading a large file.",
        "description": ("When a user uploads a large file, the application becomes unresponsive and closes "
                        "unexpectedly."),
        "steps": "1. Open the upload page.\n2. Select a large file.\n3. Click Upload.",
        "expected": ("The application should handle the file safely or display a file-size validation "
                     "message."),
        "actual": "The application crashes during upload.",
        "error": "[SAMPLE ERROR - illustrative only] Fatal error: out of memory while reading upload stream",
    },
    "UI display issue": {
        "title": "Submit button is not properly aligned on mobile screens.",
        "description": ("On smaller mobile screens, the Submit button appears outside the form container and is "
                        "difficult to access."),
        "steps": "",
        "expected": "The button should be properly aligned and accessible.",
        "actual": "The button overlaps other content.",
        "error": "",
    },
    "Minor text issue": {
        "title": "Incorrect spelling displayed in the About page heading.",
        "description": ("The About page heading displays an incorrect spelling. The rest of the application "
                        "functions normally."),
        "steps": "",
        "expected": "The heading should display the correct spelling.",
        "actual": "The heading contains a spelling mistake.",
        "error": "",
    },
}
EXAMPLE_ICONS = {"Login failure": "🔐", "Application crash": "💥", "UI display issue": "📱",
                 "Minor text issue": "✏️"}


def missing_required(values: dict) -> list[str]:
    """Labels of required form fields that are empty/whitespace."""
    return [FIELD_LABELS[k] for k in REQUIRED_FORM_FIELDS if not str(values.get(k) or "").strip()]


# ---------------------------------------------------------------- batch CSV schema
# Mirrors src.utils.TEXT_FIELDS, which SeverityPredictor.predict_batch reads.
COLUMN_GUIDE = [
    {"Column": "title", "Required": "One text column required*", "Description": "Short bug title",
     "Example": "Login fails with HTTP 500"},
    {"Column": "summary", "Required": "One text column required*",
     "Description": "One-line summary (alternative to title; Bugzilla style)", "Example": "Crash when printing"},
    {"Column": "description", "Required": "One text column required*", "Description": "What happened",
     "Example": "Valid credentials show a server error"},
    {"Column": "steps", "Required": "Optional", "Description": "Steps to reproduce",
     "Example": "1. Open login 2. Submit"},
    {"Column": "expected", "Required": "Optional", "Description": "Expected result",
     "Example": "Dashboard opens"},
    {"Column": "actual", "Required": "Optional", "Description": "Actual result", "Example": "HTTP 500 page"},
    {"Column": "error", "Required": "Optional", "Description": "Error message / stack trace",
     "Example": "500 Internal Server Error"},
]
assert [r["Column"] for r in COLUMN_GUIDE] == TEXT_FIELDS, "guide must match the prediction schema"

TEMPLATE_COLUMNS = ["title", "description", "steps", "expected", "actual", "error"]

_TEMPLATE_ROWS = [
    {"title": "Users cannot log in after clicking the Login button",
     "description": "Valid credentials show HTTP 500 Internal Server Error instead of the dashboard.",
     "steps": "1. Open Login 2. Enter valid credentials 3. Click Login",
     "expected": "The dashboard opens", "actual": "HTTP 500 error page", "error": "500 Internal Server Error"},
    {"title": "Profile picture does not update after upload",
     "description": "After uploading a new profile picture the old picture is still shown, even after reload.",
     "steps": "1. Open Profile 2. Upload a new picture 3. Reload the page",
     "expected": "The new picture is shown", "actual": "The old picture is shown", "error": ""},
    {"title": "Typo in the Settings page heading",
     "description": "The heading reads 'Setings' instead of 'Settings'. Everything else works.",
     "steps": "", "expected": "Heading reads 'Settings'", "actual": "Heading reads 'Setings'", "error": ""},
]

_SAMPLE_ROWS = [
    ("DEMO-01", "Login failure", "Users cannot log in with valid credentials",
     "Clicking Login with a valid email and password shows a server error instead of the dashboard.",
     "1. Open Login 2. Enter valid credentials 3. Click Login", "User reaches the dashboard",
     "HTTP 500 error page", "500 Internal Server Error at /api/auth/login (sample)"),
    ("DEMO-02", "Application crash", "Application crashes when exporting a large report",
     "Exporting a report with more than 10,000 rows freezes the window and the application closes.",
     "1. Open Reports 2. Select a large report 3. Click Export", "Report is exported",
     "Application closes unexpectedly", "Segmentation fault (sample)"),
    ("DEMO-03", "API response error", "Orders API returns malformed JSON",
     "The /orders endpoint sometimes returns truncated JSON, so the order list fails to load.",
     "1. Call GET /orders 2. Parse the response", "Valid JSON list of orders",
     "Truncated JSON body", "JSONDecodeError: Unterminated string (sample)"),
    ("DEMO-04", "Payment processing failure", "Card payment fails at checkout",
     "Customers see 'Payment failed' for valid cards and the order is not created.",
     "1. Add item to cart 2. Checkout 3. Pay with a valid card", "Payment succeeds and order is created",
     "'Payment failed' message", "Gateway timeout (sample)"),
    ("DEMO-05", "Broken navigation link", "Help link in the footer leads to a 404 page",
     "The Help link in the footer opens a page that does not exist.",
     "1. Scroll to footer 2. Click Help", "Help page opens", "404 Not Found page", ""),
    ("DEMO-06", "UI alignment issue", "Submit button overlaps text on mobile screens",
     "On narrow screens the Submit button overlaps the terms text below the form.",
     "1. Open the form on a phone", "Button is aligned below the text", "Button overlaps the text", ""),
    ("DEMO-07", "Incorrect spelling", "Spelling mistake on the About page",
     "The About page heading says 'Abuot us'. The rest of the page is fine.",
     "", "Heading reads 'About us'", "Heading reads 'Abuot us'", ""),
    ("DEMO-08", "Image not loading", "Product images do not load on the catalogue page",
     "Product thumbnails show a broken-image icon; product names and prices load normally.",
     "1. Open Catalogue", "Thumbnails are shown", "Broken-image icons", "GET /img/123.jpg 403 (sample)"),
    ("DEMO-09", "Search functionality issue", "Search returns no results for existing products",
     "Searching for an exact product name returns 'No results' although the product exists.",
     "1. Type an existing product name 2. Press Enter", "Matching product is listed", "'No results' message",
     ""),
    ("DEMO-10", "Form validation issue", "Registration form accepts an invalid email address",
     "The registration form accepts 'user@' as an email address and creates the account.",
     "1. Open Register 2. Enter 'user@' as email 3. Submit", "Validation message is shown",
     "Account is created", ""),
]
SAMPLE_COLUMNS = ["report_id", "issue_type", "title", "description", "steps", "expected", "actual", "error"]


def template_dataframe() -> pd.DataFrame:
    return pd.DataFrame(_TEMPLATE_ROWS, columns=TEMPLATE_COLUMNS)


def sample_reports_dataframe() -> pd.DataFrame:
    """10 demonstration reports. ``report_id`` / ``issue_type`` are extra columns that
    the prediction keeps unchanged (shows column preservation). No severity column."""
    return pd.DataFrame(_SAMPLE_ROWS, columns=SAMPLE_COLUMNS)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 with BOM so Excel on Windows opens non-ASCII characters correctly."""
    return df.to_csv(index=False).encode("utf-8-sig")


TEMPLATE_FILENAME = "bug_reports_template.csv"
SAMPLE_FILENAME = "sample_bug_reports.csv"
RESULTS_FILENAME = "bug_severity_predictions.csv"
MAX_ROWS = 5000  # practical limit for a laptop; larger files should be split


# ---------------------------------------------------------------- upload inspection
def inspect_upload(df: pd.DataFrame) -> dict:
    """Validate an uploaded frame the same way SeverityPredictor.predict_batch will
    (column names are matched case-insensitively). Returns counts and problems;
    raises nothing so the UI can show everything at once."""
    cols = [str(c).strip().lower() for c in df.columns]
    present = [c for c in TEXT_FIELDS if c in cols]
    out = {"total": int(len(df)), "text_columns": present,
           "other_columns": [c for c in cols if c not in TEXT_FIELDS],
           "missing_required": not present, "valid": 0, "invalid": 0, "invalid_rows": [],
           "duplicates": 0, "duplicate_rows": [], "too_many_rows": len(df) > MAX_ROWS}
    if not present or df.empty:
        return out
    low = df.copy()
    low.columns = cols
    texts = [combine_fields(r) for r in low[present].to_dict("records")]
    for i, t in enumerate(texts):
        try:
            validate_report_text(t)
            out["valid"] += 1
        except BugSeverityError:
            out["invalid"] += 1
            out["invalid_rows"].append(i + 2)  # spreadsheet row number (header = row 1)
    s = pd.Series([t.strip().lower() for t in texts])
    dup = s.duplicated() & (s != "")
    out["duplicates"] = int(dup.sum())
    out["duplicate_rows"] = [int(i) + 2 for i in s.index[dup]]
    return out


def restore_original_headers(result: pd.DataFrame, original_columns) -> pd.DataFrame:
    """predict_batch lower-cases header names; put the user's original spelling back
    for the input columns (the appended prediction columns keep their names)."""
    original_columns = list(original_columns)
    out = result.copy()
    out.columns = original_columns + list(result.columns[len(original_columns):])
    return out


def report_preview(df: pd.DataFrame, length: int = 90) -> list[str]:
    """Short text for the 'Bug Report' column: first non-empty text field."""
    low = df.copy()
    low.columns = [str(c).strip().lower() for c in df.columns]
    present = [c for c in TEXT_FIELDS if c in low.columns]
    out = []
    for r in low[present].to_dict("records"):
        t = combine_fields(r)
        out.append(t if len(t) <= length else t[: length - 1] + "…")
    return out


def results_csv(df_results: pd.DataFrame) -> bytes:
    """Results download: original columns + predictions. ``confidence`` is dropped
    when the model produced no probabilities (all empty), so nothing is invented."""
    out = df_results
    if "confidence" in out and out["confidence"].isna().all():
        out = out.drop(columns="confidence")
    return to_csv_bytes(out)


def read_upload(data: bytes) -> pd.DataFrame:
    from src.preprocessing import read_csv
    return read_csv(io.BytesIO(data))
