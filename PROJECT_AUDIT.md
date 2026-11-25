# Project Audit – original `ozsaygin/severity-prediction`

Audit performed on the unmodified clone (branch `master`, HEAD `2a2515e "Added research documentations"`, 6 commits).

## 1. Files that actually exist

| File | Purpose |
|---|---|
| `train_and_validate.py` | Preprocess, train Word2Vec + MLP, 5-fold validation, pickle models |
| `eval.py` | Interactive CLI: type a summary, get a severity |
| `data_helpers.py` | NLTK-based text helpers (lowercase, punctuation, tokenize, stopwords, stemming) |
| `summaryList.csv` | **Real** Bugzilla sample: 99 verified bug summaries |
| `nn_model.bin`, `w2v_model.bin` | Pre-trained pickles (MLP, gensim Word2Vec) |
| `docs/*.ipynb`, `docs/*.pdf` | Research notebook (79 cells) and report |
| `README.md`, `LICENSE` (MIT) | Documentation |

No `requirements.txt`, no tests, no `.gitignore` (a macOS `.DS_Store` is committed).

## 2. Original architecture & algorithm
`CSV → lowercase → strip punctuation → nltk word_tokenize → gensim Word2Vec (100-dim, 1000 epochs, trained on the same 99 summaries) → mean-embedding vector per summary → sklearn MLPClassifier (1 hidden layer of 100 units, tanh, lbfgs) → label`.
Evaluation: 5-fold `StratifiedKFold`, accuracy only. The final model is retrained on all data and pickled.

## 3. Dataset structure
CSV with single-quote quoting (`quotechar="'"`), columns `summary, severity, status, assigned_to, bug_id`.
Only `summary` (feature) and `severity` (label) are used by the code; `status`, `assigned_to`, `bug_id` are ignored.

Measured distribution of `severity` (99 rows, 0 duplicates, 0 missing, all `VERIFIED`):

| raw label | count |
|---|---|
| normal | 49 |
| critical | 35 |
| enhancement | 6 |
| major | 4 |
| minor | 2 |
| trivial | 2 |
| blocker | 1 |

The README promises 3 classes (`non-critical`, `normal`, `critical`) but the CSV contains **7 raw Bugzilla labels**. The original code never maps them, so the MLP actually trained on 7 classes, several with ≤2 examples. `class_names` in the script is defined but unused (and contains a stray space in `" severe"`).

## 4. Bugs and compatibility problems found
1. `pd.read_csv(..., error_bad_lines=False)` – removed in pandas 2.0 (crash).
2. `gensim.models.Word2Vec(size=, iter=)` – renamed `vector_size`/`epochs` in gensim 4 (crash).
3. `model.wv.index2word`, `model.wv.syn0`, `model.wv.vocab` – removed in gensim 4.
4. `workers=-1` is invalid for Word2Vec.
5. `from sklearn.model_selection import *` / `sklearn.metrics import *` – wildcard imports.
6. `StratifiedKFold` is used with classes of 1–2 samples → warnings/unreliable folds; no shuffling.
7. `w2v_model.bin` is first written by `save_word2vec_format` and then **overwritten** by `pickle.dump(model)` – two incompatible formats under one name.
8. `MeanEmbeddingVectorizer` is copy-pasted in two files; divides by `len(words)` (crashes with ZeroDivisionError on empty input), hard-codes 100 dims.
9. `remove_stopwords` rebuilds `set(stopwords.words("english"))` for every token (very slow); the stopword and stemming steps are commented out in training anyway.
10. Punctuation stripping deletes `_ . : / 0x…` characters, destroying exactly the technical tokens (`JS_Assert`, `0x1012279c`, file paths) that matter in bug reports.
11. Hard-coded Anaconda shebang (`#!/anaconda3/envs/...`) – unusable on Windows.
12. `pip install ... pickle` in the README is wrong (`pickle` is stdlib). `LICENCE` link is broken (file is `LICENSE`).
13. NLTK data (`punkt`, `stopwords`) is never downloaded by the code.
14. Validation and final training use the same data and report only accuracy; there is no held-out test set, no per-class metrics, no confusion matrix.
15. `df.values[id][0]` positional column access – breaks if column order changes.

## 5. Missing features
Input validation, label normalisation, tests, GUI, batch prediction, probabilities, reproducible seeds, model metadata, documentation of results, `requirements.txt`.

## 6. Security / maintenance concerns
* `pickle.load` on `nn_model.bin` / `w2v_model.bin` executes arbitrary code if a file is tampered with. The pre-trained pickles were also produced by Python 3.6-era sklearn/gensim and are not reliably loadable today. **We do not load them**; they are retained untouched for history and the model is retrained.
* Dependencies unpinned; Python "> 3.0" claim is untrue for any modern dependency set.
* Dataset is tiny (99 rows): any metric from it is high-variance.
* Dataset contains developer email addresses (`assigned_to`); these are not used and are not shown in the new app.

## 7. Modernisation decisions
* Default features: **TF-IDF** (word + char n-grams) – lightweight, keeps technical tokens. Word2Vec mean-embedding is kept as an optional comparison only if `gensim` installs on the chosen Python.
* Models compared on one shared split via cross-validation: Logistic Regression, Multinomial NB, Linear SVM, Random Forest.
* Explicit label map: `critical, blocker, major → Critical`; `normal → Normal`; `minor, trivial, enhancement → Non-Critical`. (`enhancement` is a feature request rather than a defect; mapping it to Non-Critical is a documented judgement call.)
* Original files are left in place unmodified.

## 8. Follow-up findings (after modernisation)

* **Dataset provenance verified.** All 99 `bug_id`s were looked up in the public Bugzilla REST API: all 99 summaries and severity labels match the live records (Core / Core Graveyard / Firefox Build System, status `VERIFIED`, created 2007-01-02 … 2007-02-11). Bugzilla now stores the 6 `enhancement` rows as `type=enhancement, severity=normal`.
* **One exact duplicate.** Bugs 367589 and 369404 differ only by quote characters; the current loader removes such duplicates (99 → 98 rows). The shipped model was trained before this change, on all 99 rows.
* **Independent check of the shipped model.** On 793 further real, held-out Bugzilla reports (same products/status, 2006–2008, collected with `scripts/collect_bugzilla.py`, none of the original 99), the shipped Random Forest scored accuracy 0.733, macro-F1 0.528 and found 2 of 56 Non-Critical reports (`reports/experiments.json`, stage s3). The 0.80 accuracy on the 20-row test split was therefore optimistic.
* **`major → Critical` is questionable on this data.** Out-of-fold, 263 of 314 `major` reports were predicted *Normal* (stage s5). The mapping was not changed; it is documented as a judgement call.
* **CSV robustness fixes.** Excel's Windows-1252 "CSV" files failed to load, single-column CSVs were re-read with `'` as quote character (corrupting apostrophes), rows with unquoted commas silently lost data, and renamed Excel workbooks gave a misleading error. All four are fixed and covered by tests.
