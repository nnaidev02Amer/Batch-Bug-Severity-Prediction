"""Fixed, leakage-safe evaluation splits shared by training and experiments.

Splits (all deterministic for a given seed):

* ``orig_test``  - the SAME 20 rows of the original 99-row sample that the previous
  version held out (stratified 80/20, seed 42, computed on the raw file). Keeping
  them fixed makes old and new results directly comparable.
* ``orig_train`` - the remaining original rows (exact duplicates removed; a training
  row that is a near-duplicate of a test row is dropped).
* ``ext_test``   - ~20 % of the additionally collected Bugzilla reports, chosen with a
  stratified *group* split so near-duplicate reports never straddle train/test.
* ``ext_train``  - the rest of the collected reports.

Collected reports whose bug_id is in the original sample, or which are
near-duplicates of an ``orig_test`` row, are excluded from every training set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .preprocessing import combine_fields, load_training_data, near_duplicate_groups, read_csv
from .utils import DEFAULT_DATA, EXTRA_DATA

NEAR_DUP_THRESHOLD = 0.8


def original_test_ids(path: str | Path = DEFAULT_DATA, seed: int = 42, test_size: float = 0.2) -> set[int]:
    """bug_ids of the previous version's held-out rows (split on the raw file,
    exactly as the previous ``train()`` did before duplicate handling changed)."""
    from .preprocessing import normalize_label
    raw = read_csv(path)
    y = [normalize_label(s) for s in raw["severity"]]
    _, te = train_test_split(raw["bug_id"].to_numpy(), test_size=test_size, stratify=y,
                             random_state=seed)
    return {int(i) for i in te}


def build_splits(orig_path: str | Path = DEFAULT_DATA, extra_path: str | Path | None = EXTRA_DATA,
                 seed: int = 42, test_size: float = 0.2) -> dict:
    """Return {'orig_train','orig_test','ext_train','ext_test'} DataFrames (columns
    text,label,raw_severity,bug_id,group) plus an 'info' dict."""
    orig, orig_info = load_training_data(orig_path)
    test_ids = original_test_ids(orig_path, seed, test_size)
    info = {"orig_data_info": orig_info, "seed": seed, "near_dup_threshold": NEAR_DUP_THRESHOLD}

    frames = [orig.assign(source="orig")]
    ext_info = None
    if extra_path is not None and Path(extra_path).exists():
        ext, ext_info = load_training_data(extra_path)
        n0 = len(ext)
        ext = ext[~ext["bug_id"].isin(orig["bug_id"])]
        info["ext_removed_in_original_sample"] = n0 - len(ext)
        frames.append(ext.assign(source="ext"))
    info["ext_data_info"] = ext_info

    allrows = pd.concat(frames, ignore_index=True)
    allrows["group"] = near_duplicate_groups(allrows["text"], NEAR_DUP_THRESHOLD)
    is_otest = (allrows["source"] == "orig") & allrows["bug_id"].isin(test_ids)
    tainted = set(allrows.loc[is_otest, "group"])
    leak = ~is_otest & allrows["group"].isin(tainted)
    info["removed_near_dup_of_orig_test"] = {
        "orig": int((leak & (allrows["source"] == "orig")).sum()),
        "ext": int((leak & (allrows["source"] == "ext")).sum())}

    out = {"orig_test": allrows[is_otest],
           "orig_train": allrows[(allrows["source"] == "orig") & ~is_otest & ~leak]}
    ext = allrows[(allrows["source"] == "ext") & ~leak]
    if len(ext):
        n_splits = max(2, round(1 / test_size))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tr, te = next(cv.split(np.zeros(len(ext)), ext["label"], ext["group"]))
        ext_tr, ext_te = ext.iloc[tr], ext.iloc[te]
        # A test group that also contains an original TRAINING row moves to training.
        move = ext_te["group"].isin(set(out["orig_train"]["group"]))
        info["ext_test_moved_to_train_near_dup_of_orig_train"] = int(move.sum())
        out["ext_train"] = pd.concat([ext_tr, ext_te[move]])
        out["ext_test"] = ext_te[~move]
    else:
        out["ext_train"] = out["ext_test"] = ext
    out = {k: v.reset_index(drop=True) for k, v in out.items()}
    info["sizes"] = {k: int(len(v)) for k, v in out.items()}
    info["class_counts"] = {k: v["label"].value_counts().to_dict() for k, v in out.items()}
    out["info"] = info
    return out
