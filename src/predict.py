"""Load the saved model and predict severity for single or batch bug reports."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .modeling import align_proba, decide
from .preprocessing import combine_fields, read_csv, validate_report_text
from .utils import (CLASSES, DEFAULT_MODEL, LOW_CONFIDENCE_MESSAGE, PRIORITY, TEXT_FIELDS,
                    BugSeverityError, ModelNotFoundError)

DISCLAIMER = ("Automated recommendation only. It reflects patterns in historical Bugzilla "
              "reports, not business impact; a human should confirm before acting.")
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


class SeverityPredictor:
    def __init__(self, path: str | Path = DEFAULT_MODEL):
        path = Path(path)
        if not path.exists():
            raise ModelNotFoundError(
                f"Model file not found: {path}. Train one with `python -m src.train`.")
        try:
            # Only load model files you trained yourself: joblib uses pickle.
            bundle = joblib.load(path)
            self.pipeline = bundle["pipeline"]
            self.metadata = bundle["metadata"]
            if not hasattr(self.pipeline, "predict"):
                raise TypeError("stored 'pipeline' object cannot predict")
        except (KeyError, TypeError, EOFError, ValueError, AttributeError) as exc:
            raise BugSeverityError(f"Model file {path} is corrupt or incompatible: {exc}") from exc
        # Older bundles have no threshold / decision priors: fall back to defaults.
        self.threshold = float(self.metadata.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
        self.priors = self.metadata.get("decision_priors")

    # -- scoring -----------------------------------------------------------
    def _scores(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (labels, probabilities in CLASSES order, or None if the model
        has no probability output, e.g. an uncalibrated Linear SVM)."""
        if not hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict(texts), None
        proba = align_proba(self.pipeline, self.pipeline.predict_proba(texts))
        if self.priors:  # decision rule chosen during training (see MODEL_REPORT.md)
            return decide(proba, self.priors), proba
        return self.pipeline.predict(texts), proba

    def is_low_confidence(self, confidence) -> bool | None:
        """True if the probability of the predicted class is below the threshold.
        None when the model provides no probability (unknown, treat as uncertain)."""
        if confidence is None or (isinstance(confidence, float) and np.isnan(confidence)):
            return None
        return bool(confidence < self.threshold)

    def predict_one(self, **fields) -> dict:
        """Predict from named fields (title, description, steps, expected, actual, error)."""
        text = combine_fields(fields)
        validate_report_text(text)
        labels, proba = self._scores([text])
        label = str(labels[0])
        out = {"severity": label, "priority": PRIORITY[label], "disclaimer": DISCLAIMER,
               "probabilities": None, "confidence": None, "low_confidence": None,
               "review_note": LOW_CONFIDENCE_MESSAGE, "threshold": self.threshold}
        if proba is not None:
            out["probabilities"] = {c: float(p) for c, p in zip(CLASSES, proba[0])}
            out["confidence"] = out["probabilities"][label]
            out["low_confidence"] = self.is_low_confidence(out["confidence"])
            if not out["low_confidence"]:
                out["review_note"] = ""
        return out

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add predicted_severity / priority / confidence columns. Rows with no
        usable text get 'Invalid' rather than failing the whole batch."""
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        present = [c for c in TEXT_FIELDS if c in df.columns]
        if not present:
            raise BugSeverityError(
                f"CSV must contain at least one text column from: {', '.join(TEXT_FIELDS)}. "
                f"Found columns: {list(df.columns)}")
        if df.empty:
            raise BugSeverityError("The CSV has no rows.")
        texts = [combine_fields(r) for r in df[present].to_dict("records")]
        valid = []
        for t in texts:
            try:
                validate_report_text(t)
                valid.append(True)
            except BugSeverityError:
                valid.append(False)
        sev = ["Invalid (empty text)"] * len(df)
        pri = [""] * len(df)
        conf = [np.nan] * len(df)
        idx = [i for i, v in enumerate(valid) if v]
        if idx:
            labels, proba = self._scores([texts[i] for i in idx])
            for k, i in enumerate(idx):
                sev[i] = str(labels[k])
                pri[i] = PRIORITY[sev[i]]
                if proba is not None:
                    conf[i] = float(proba[k][CLASSES.index(sev[i])])
        df["predicted_severity"] = sev
        df["priority"] = pri
        df["confidence"] = conf
        df["review"] = ["" if not v else
                        (LOW_CONFIDENCE_MESSAGE if self.is_low_confidence(c) in (True, None) else "")
                        for v, c in zip(valid, conf)]
        return df

    def predict_csv(self, path_or_buffer) -> pd.DataFrame:
        return self.predict_batch(read_csv(path_or_buffer))
