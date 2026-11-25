# Data

| File | Origin | Real or synthetic |
|---|---|---|
| `bugzilla_summary_sample.csv` | Byte-identical copy of `summaryList.csv` from the upstream repo `ozsaygin/severity-prediction` (the original at the repo root is never modified) | **Real**, 99 rows |
| `bugzilla_core_verified_2006_2008.csv` | Collected with `scripts/collect_bugzilla.py` from the public Mozilla Bugzilla REST API (`https://bugzilla.mozilla.org/rest/bug`), retrieved 2026-11-25; query and timestamp in `bugzilla_core_verified_2006_2008.provenance.json` | **Real**, 4,079 rows |
| `sample_batch_upload.csv` | First 15 rows of the original sample (`bug_id`, `summary` only) for trying *Batch Prediction* | **Real** |

The tests use a small **synthetic** dataset generated in `tests/conftest.py`; it is never used for the shipped model or any reported metric. No synthetic rows exist in any file in this folder.

## Provenance of the original sample (verified 2026-11-25)

All 99 `bug_id`s were looked up in the live Bugzilla API: **all 99 summaries and all 99 severity labels match** the live records. They are products Core (95) / Core Graveyard (2) / Firefox Build System (2), status `VERIFIED`, created 2007-01-02 … 2007-02-11. Bugzilla has since moved `enhancement` from the *severity* field to a separate *type* field: the 6 `enhancement` rows are now `severity=normal, type=enhancement`.

## Collected data (`bugzilla_core_verified_2006_2008.csv`)

* **Selection** mirrors the original sample: products Core / Core Graveyard / Firefox Build System, status `VERIFIED`, created 2006-01-01 … 2008-12-31. The collector re-finds all 99 original bugs (flag `in_original_sample = 1`); they are excluded from the extended training/test sets so nothing is counted twice.
* **Columns:** `bug_id, summary, severity, label_source, severity_live, type_live, status, resolution, product, component, creation_time, in_original_sample`.
* **Labels are not invented.** `severity_live`/`type_live` are exactly what the API returns. `severity` restores the original vocabulary with one documented rule: `enhancement` if `type_live = enhancement` and `severity_live = normal` (as verified on the 6 original enhancement rows), otherwise `severity_live` unchanged (`label_source` records which).
* **Unknown labels** (3 rows with post-2020 `S3`/`S4`) are kept in the file but dropped and counted by the training loader.
* **Privacy:** no e-mail address (`assigned_to`, reporter) is requested or stored.
* **Quality** (measured): 0 missing values, 0 duplicate ids; 10 exact duplicate summaries (after case/space normalisation) and 3 identical summaries with *conflicting* labels (6 rows) are removed by the loader. Some rows are junk (e.g. "Test", a bare URL, spam about a "membership registration page"); they are real labelled records and were kept.
* **Reproduce:** `python scripts/collect_bugzilla.py` (Bugzilla data can change slightly over time, e.g. bugs re-triaged or hidden; the committed CSV is the snapshot used for all reported results).

## Training CSV format

Single- or double-quoted CSV with a text column (`summary`, or `title`/`description`) and a `severity` column. Other columns are ignored (`bug_id` is kept for de-duplication).

Severity labels (case-insensitive) are mapped as:

| Raw label | Class |
|---|---|
| critical, blocker, major, severe | Critical |
| normal, medium | Normal |
| minor, trivial, enhancement, non-critical, non-severe, low | Non-Critical |

Rows with other labels (e.g. `S1`–`S4`) are dropped and counted in the training log / report, never guessed.

## Batch prediction CSV format

At least one of: `title`, `summary`, `description`, `steps`, `expected`, `actual`, `error`. Extra columns are preserved. Output adds `predicted_severity`, `priority`, `confidence` and `review` (`Low confidence – Manual review recommended.` when the calibrated probability is below the threshold).
