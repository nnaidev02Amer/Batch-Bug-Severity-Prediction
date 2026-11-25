"""CSV loading, label normalisation and text cleaning.

Text cleaning deliberately keeps technical tokens (``js_assert``, ``0x1012279c``,
``nsIDocShell.cpp:42``, file paths) instead of stripping all punctuation as the
original project did.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .utils import CLASSES, LABEL_MAP, TEXT_FIELDS, BugSeverityError

# Negations carry meaning in bug reports ("does not start"): never drop them.
KEEP_WORDS = {"not", "no", "nor", "cannot", "never", "without", "fail", "fails"}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS - KEEP_WORDS)

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_NON_TOKEN = re.compile(r"[^\w.:/\\\-+#]+")
# Token = word chars optionally joined by . : / \ - + (keeps identifiers & codes)
TOKEN_PATTERN = r"(?u)\b\w[\w.:/\\\-+#]*\w\b|\b\w\b"
_TOKEN_RE = re.compile(TOKEN_PATTERN)


def normalize_label(raw) -> str:
    """Map a raw severity label to Critical / Normal / Non-Critical.

    Raises BugSeverityError for unknown or missing labels.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        raise BugSeverityError("Missing severity label.")
    key = re.sub(r"[\s_]+", "-", str(raw).strip().lower())
    if key not in LABEL_MAP:
        raise BugSeverityError(
            f"Unknown severity label {raw!r}. Known labels: {sorted(LABEL_MAP)}"
        )
    return LABEL_MAP[key]


def clean_text(text) -> str:
    """Lower-case, replace URLs/emails with placeholders, keep technical tokens."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text).lower()
    t = _URL.sub(" urltoken ", t)
    t = _EMAIL.sub(" emailtoken ", t)
    t = _NON_TOKEN.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


# --- alternative normalisers (compared in src/experiments.py) -------------
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_IDENT = re.compile(r"[A-Za-z_][\w:.]*\w")
_PUNCT = re.compile(r"[^a-z0-9\s]+")


def split_identifier(token: str) -> list[str]:
    """``nsThebesImage::AllowedImageSize`` -> ns, Thebes, Image, Allowed, Image, Size."""
    out = []
    for part in re.split(r"[_:.]+", token):
        out += [p for p in _CAMEL.split(part) if p]
    return out


def expand_identifiers(text) -> str:
    """Keep the text unchanged but append the sub-words of compound identifiers
    (camelCase, snake_case, ``a::b``, ``a.b``) and mark Mozilla crash signatures
    (``[@ frame]``) with the token ``crashsignaturetoken``. The original tokens are
    never removed, so ``JS_Assert`` still exists as a whole token as well."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text)
    extra = []
    for m in _IDENT.finditer(t):
        parts = split_identifier(m.group())
        if len(parts) > 1:
            extra.extend(parts)
    t = t.replace("[@", " crashsignaturetoken [")
    return t + (" " + " ".join(extra) if extra else "")


def clean_text_split(text) -> str:
    """``clean_text`` + identifier sub-words + crash-signature marker."""
    return clean_text(expand_identifiers(text))


def clean_text_basic(text) -> str:
    """Original-project style: lower-case and strip ALL punctuation. Only used as
    a comparison baseline; it destroys tokens such as ``0x1012279c:42``."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", str(text).lower())).strip()


NORMALIZERS = {"basic": clean_text_basic, "technical": clean_text,
               "technical+split": clean_text_split}


def near_duplicate_groups(texts, threshold: float = 0.8, chunk: int = 500):
    """Group reports whose character-n-gram TF-IDF cosine similarity >= threshold
    (connected components). Used so near-duplicates never straddle train/test."""
    import numpy as np
    from scipy.sparse import vstack
    from scipy.sparse.csgraph import connected_components
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = list(texts)
    if len(texts) < 2:
        return np.zeros(len(texts), dtype=int)
    X = TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb",
                        ngram_range=(3, 5)).fit_transform(texts)
    blocks = []
    for start in range(0, X.shape[0], chunk):  # chunked to bound memory
        S = (X[start:start + chunk] @ X.T).tocsr()
        S.data[S.data < threshold] = 0
        S.eliminate_zeros()
        blocks.append(S)
    _, labels = connected_components(vstack(blocks), directed=False)
    return labels


def tokenize(text: str) -> list[str]:
    """Tokenise cleaned text and drop stopwords (negations are kept)."""
    return [tok for tok in _TOKEN_RE.findall(text) if tok not in STOPWORDS]


def combine_fields(row: dict) -> str:
    """Join all available text fields of one bug report into a single string."""
    parts = []
    for key in TEXT_FIELDS:
        val = row.get(key)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            val = str(val).strip()
            if val:
                parts.append(val)
    return " . ".join(parts)


def validate_report_text(text: str) -> str:
    """Return cleaned text or raise if the report is effectively empty."""
    cleaned = clean_text(text)
    if not tokenize(cleaned):
        raise BugSeverityError(
            "The bug report is empty (or contains only stop-words/symbols). "
            "Please provide a title or description."
        )
    return cleaned


def _read_bytes(path_or_buffer) -> bytes:
    if hasattr(path_or_buffer, "read"):
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        data = path_or_buffer.read()
        return data.encode("utf-8") if isinstance(data, str) else data
    try:
        return Path(path_or_buffer).read_bytes()
    except (FileNotFoundError, IsADirectoryError):
        raise BugSeverityError(f"CSV file not found: {path_or_buffer}") from None


def decode_csv_bytes(data: bytes) -> tuple[str, str]:
    """Decode CSV bytes: UTF-8 (with or without the BOM Excel's 'CSV UTF-8' adds),
    falling back to Windows-1252 (Excel's plain 'CSV' on Windows). Returns (text, encoding)."""
    if data[:4] == b"PK\x03\x04" or data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        raise BugSeverityError("This looks like an Excel workbook (.xlsx/.xls), not a CSV file. "
                               "In Excel use File > Save As > 'CSV UTF-8 (Comma delimited)' and upload that.")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise BugSeverityError("Could not decode the file. Please save it as 'CSV UTF-8'.")


def read_csv(path_or_buffer) -> pd.DataFrame:
    """Read a CSV (path or file-like). Handles UTF-8/BOM and Windows-1252 files and
    the single-quote style of the original ``summaryList.csv`` (detected from the
    header line). Raises BugSeverityError on missing, empty or unreadable files."""
    import io
    text, _ = decode_csv_bytes(_read_bytes(path_or_buffer))
    if not text.strip():
        raise BugSeverityError("The CSV file is empty.")
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    quote = "'" if first.lstrip().startswith("'") else '"'
    import warnings
    try:
        with warnings.catch_warnings():
            # Rows with more values than headers would silently lose data: make it an error.
            warnings.simplefilter("error", pd.errors.ParserWarning)
            df = pd.read_csv(io.StringIO(text), quotechar=quote, index_col=False)
    except pd.errors.ParserWarning:
        raise BugSeverityError(
            "Some rows have more values than there are column headers. Text that contains commas "
            "must be wrapped in double quotes (Excel does this automatically when saving as CSV).") from None
    except pd.errors.EmptyDataError:
        raise BugSeverityError("The CSV file is empty.") from None
    except pd.errors.ParserError as exc:
        raise BugSeverityError(f"Could not parse CSV file: {exc}") from None
    df.columns = [str(c).strip().strip("'\"").lower() for c in df.columns]
    return df


def load_training_data(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load, validate and clean a labelled dataset.

    Returns (dataframe with columns ``text`` and ``label``, info dict describing
    what was dropped). Requires a ``summary`` (or ``title``/``description``)
    column and a ``severity`` column.
    """
    df = read_csv(path)
    if "severity" not in df.columns:
        raise BugSeverityError("Training CSV needs a 'severity' column.")
    if not any(c in df.columns for c in ("summary", "title", "description")):
        raise BugSeverityError(
            "Training CSV needs a text column: 'summary', 'title' or 'description'."
        )
    info = {"rows_raw": int(len(df))}

    df["text"] = [combine_fields(r) for r in df.to_dict("records")]
    df["raw_severity"] = df["severity"].astype(str).str.strip().str.lower()

    before = len(df)
    df = df[df["text"].map(lambda t: bool(tokenize(clean_text(t))))]
    info["dropped_empty_text"] = before - len(df)

    # Unknown labels are dropped (and counted), not guessed.
    labels, unknown = [], {}
    for raw in df["severity"]:
        try:
            labels.append(normalize_label(raw))
        except BugSeverityError:
            labels.append(None)
            unknown[str(raw)] = unknown.get(str(raw), 0) + 1
    df["label"] = labels
    info["dropped_unknown_label"] = unknown
    df = df[df["label"].notna()]

    # Exact duplicates are judged on the *cleaned* text (case/spacing variants).
    before = len(df)
    df = (df.assign(_k=df["text"].map(clean_text))
            .drop_duplicates(subset=["_k", "label"]).drop(columns="_k"))
    info["dropped_duplicates"] = before - len(df)

    # Identical text carrying different labels cannot be learned: drop all copies.
    key = df["text"].map(clean_text)
    conflicting = df.assign(k=key).groupby("k")["label"].transform("nunique") > 1
    info["dropped_conflicting_labels"] = int(conflicting.sum())
    df = df[~conflicting]

    df = df.reset_index(drop=True)
    if df.empty:
        raise BugSeverityError("No usable rows left after validation.")
    info["rows_used"] = int(len(df))
    info["raw_label_counts"] = df["raw_severity"].value_counts().to_dict()
    info["class_counts"] = {c: int((df["label"] == c).sum()) for c in CLASSES}
    cols = ["text", "label", "raw_severity"] + (["bug_id"] if "bug_id" in df.columns else [])
    return df[cols], info
