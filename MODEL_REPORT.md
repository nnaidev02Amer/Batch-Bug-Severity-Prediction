# Model Report

*Generated automatically by `src/train.py` on 2026-11-25T06:22:10+00:00 (UTC).*

> **Read this first:** the dataset is tiny (see below). Metrics have very high variance and must not be read as real-world performance. This is a demonstration system, not a production triage tool.

## Data

* Source: `data/bugzilla_summary_sample.csv` – real Bugzilla bug summaries shipped with the upstream project (not synthetic).
* Rows in file: 99; used after validation: 99
* Dropped – empty text: 0, duplicates: 0, unknown labels: none
* Raw labels: {'normal': 49, 'critical': 35, 'enhancement': 6, 'major': 4, 'minor': 2, 'trivial': 2, 'blocker': 1}
* Mapped classes: {'Critical': 40, 'Normal': 49, 'Non-Critical': 10}
* Mapping: critical/blocker/major → Critical; normal → Normal; minor/trivial/enhancement → Non-Critical.

## Setup

* Stratified split, seed 42: 79 train / 20 test rows.
* Model selection: 5-fold × 3 repeated stratified CV on the **train split only**; vectoriser is fitted inside each fold (no leakage).
* Class imbalance handled with `class_weight='balanced'` where the estimator supports it.
* Versions: {'python': '3.14.6', 'scikit-learn': '1.9.1', 'numpy': '2.5.5'}

## Model comparison (CV on train split)

| Model | CV macro-F1 | CV accuracy |
|---|---|---|
| Random Forest **(selected)** | 0.530 ± 0.053 | 0.752 ± 0.078 |
| Logistic Regression | 0.502 ± 0.083 | 0.698 ± 0.100 |
| Linear SVM | 0.502 ± 0.083 | 0.698 ± 0.100 |
| Multinomial Naive Bayes | 0.468 ± 0.065 | 0.667 ± 0.090 |

Majority-class baseline on the test split (always 'Normal'): accuracy 0.500, macro-F1 0.222.

## Held-out test result – Random Forest (20 samples)

* Accuracy: 0.800
* Macro precision/recall/F1: 0.571 / 0.583 / 0.563
* Weighted precision/recall/F1: 0.757 / 0.800 / 0.760

```
              precision    recall  f1-score   support

    Critical       1.00      0.75      0.86         8
      Normal       0.71      1.00      0.83        10
Non-Critical       0.00      0.00      0.00         2

    accuracy                           0.80        20
   macro avg       0.57      0.58      0.56        20
weighted avg       0.76      0.80      0.76        20
```

Confusion matrix (rows = true, columns = predicted; order Critical, Normal, Non-Critical):

```
[6, 2, 0]
[0, 10, 0]
[0, 2, 0]
```

With so few test rows (and only a handful of Non-Critical examples) a single misclassification moves per-class numbers by tens of percentage points.

## Final deployed model

After selection, the winning pipeline is re-fitted on **all** rows and saved to `models/trained_model.joblib`. Its performance is estimated by the numbers above, not measured separately.

## Known limitations

* ~100 training reports, all from Mozilla Bugzilla (JavaScript/Core era, mostly mid-2000s): the model will not generalise to other products or modern writing styles.
* Severity labels in Bugzilla are subjective and inconsistently assigned; `enhancement`→Non-Critical and `major`→Critical are judgement calls.
* Word2Vec (used originally) could not be evaluated: `gensim` fails to import against the current SciPy on Python 3.14. With ~100 sentences it would not have been competitive anyway, but this was not measured.
* The Non-Critical class has only ~10 examples; check the per-class table above – a recall of 0 there means the model effectively cannot identify that class yet.
* Logistic Regression and Linear SVM often produce identical predictions on this data (high-dimensional TF-IDF, ~80 samples, both fit near-identical linear margins); equal scores are expected, not a bug.
* Predictions are recommendations for a human triager, not determinations of business impact.
