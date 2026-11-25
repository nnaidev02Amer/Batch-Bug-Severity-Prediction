"""Bug Severity Prediction System – Streamlit dashboard.  Run: streamlit run app.py

Presentation only: every prediction comes from src.predict.SeverityPredictor and every
metric from the metadata saved with the trained model. Nothing here is hard-coded.
"""
from __future__ import annotations

import hashlib
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import SeverityPredictor
from src.preprocessing import load_training_data
from src.utils import (CLASSES, DEFAULT_DATA, DEFAULT_MODEL, EXTRA_DATA, LOW_CONFIDENCE_MESSAGE,
                       PRIORITY, BugSeverityError, ModelNotFoundError)
from ui import content as C
from ui.theme import (COLORS, ICONS, card, hero, inject_css, kpi, section, severity_result,
                      sidebar_brand, steps, style_fig)

st.set_page_config(page_title="Bug Severity Prediction", page_icon="🐞", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()

EXPERIMENTAL_WARNING = ("This prediction is experimental. The model was trained on a limited dataset. "
                        "Please verify the severity manually before prioritizing the issue.")
PAGE_LABELS = ["🏠 Home", "🔮 Predict", "📄 Batch Prediction", "📊 Analytics", "🧠 Model Info", "ℹ️ About"]
FORM_KEYS = {f: f"f_{f}" for f in C.FORM_FIELDS}


# ---------------------------------------------------------------- data access
@st.cache_resource(show_spinner="Loading model…")
def get_predictor():
    return SeverityPredictor(DEFAULT_MODEL)


@st.cache_data(show_spinner=False)
def get_datasets():
    """(original sample, its info), (collected reports or None, info or None)."""
    orig = load_training_data(DEFAULT_DATA)
    ext = None
    if EXTRA_DATA.exists():
        df, info = load_training_data(EXTRA_DATA)
        ext = (df[~df["bug_id"].isin(orig[0]["bug_id"])], info)
    return orig, ext


def load_or_stop():
    try:
        return get_predictor()
    except (ModelNotFoundError, BugSeverityError) as exc:
        st.error(f"**The trained model could not be loaded.** {exc}")
        st.code("python -m src.train", language="bash")
        st.stop()


def model_uses_collected_data(meta: dict) -> bool:
    return "ext_test" in meta.get("test_sets", {})


def confidence_label(meta: dict) -> str:
    cal = meta.get("calibration")
    return "calibrated probability" if cal and cal != "none" else "uncalibrated model score"


def go_to(page: str):
    st.session_state["nav"] = page


def bar(counts: dict, title: str, height: int = 340):
    df = pd.DataFrame({"Severity": list(counts), "Reports": list(counts.values())})
    fig = px.bar(df, x="Severity", y="Reports", color="Severity", text="Reports",
                 color_discrete_map=COLORS, title=title)
    fig.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
    return style_fig(fig, height)


# ---------------------------------------------------------------- Home
def page_home(pred):
    meta = pred.metadata
    hero("Bug Severity Prediction System",
         "Paste a bug report and get a suggested severity (Critical / Normal / Non-Critical) and priority, "
         "so likely-critical issues can be looked at first. Every suggestion is meant to be checked by a person.",
         tag="Machine learning · NLP · Triage assistant")
    c = meta["class_counts"]
    k = st.columns(4)
    with k[0]:
        kpi("Training reports", meta["n_total"], "real Mozilla Bugzilla reports")
    for col, cls in zip(k[1:], CLASSES):
        with col:
            kpi(f"{ICONS[cls]} {cls}", c[cls], f"{c[cls] / meta['n_total']:.0%} of training data", COLORS[cls])

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.plotly_chart(bar(c, "Severity distribution of the training data"), width="stretch")
    with right:
        tm = meta["test_metrics"]
        with st.container(border=True):
            st.markdown("#### 🧠 Current model")
            st.markdown(f"**{meta['algorithm']}**  \n{meta['feature_extraction']}")
            m = st.columns(2)
            m[0].metric("Held-out accuracy", f"{tm['accuracy']:.0%}")
            m[1].metric("Macro-F1", f"{tm['macro']['f1']:.2f}")
            st.caption(f"Measured on only {tm['n_samples']} held-out reports – treat as a rough indication, "
                       "not real-world performance.")
        with st.container(border=True):
            st.markdown("#### 🧭 Severity → priority")
            for cls in CLASSES:
                st.markdown(f"{ICONS[cls]} **{cls}** → {PRIORITY[cls]}")

    st.markdown("### Quick start")
    q = st.columns(3)
    with q[0]:
        with st.container(border=True):
            st.markdown("#### 🔮 Check one bug")
            st.caption("Fill in a simple form, or load an example report to see how it works.")
            st.button("Open Predict", key="qs_predict", on_click=go_to, args=("🔮 Predict",),
                      width="stretch", type="primary")
    with q[1]:
        with st.container(border=True):
            st.markdown("#### 📄 Check many bugs")
            st.caption("Download a CSV template, fill it in Excel, upload it and download the results.")
            st.button("Open Batch Prediction", key="qs_batch", on_click=go_to, args=("📄 Batch Prediction",),
                      width="stretch")
    with q[2]:
        with st.container(border=True):
            st.markdown("#### 📊 See how reliable it is")
            st.caption("Accuracy, per-class results, confusion matrix and known limitations.")
            st.button("Open Analytics", key="qs_analytics", on_click=go_to, args=("📊 Analytics",),
                      width="stretch")

    with st.expander("What is a bug report, and what does this tool do?"):
        st.markdown("""
A **bug report** describes something in a piece of software that does not work as it should: *what* went
wrong, *how* to make it happen again, what *should* have happened and what *actually* happened.

This tool reads the **text** of a report and suggests how severe it is, based on patterns learned from
real, historical Mozilla Bugzilla reports:

* 🔴 **Critical** – crashes, hangs, data loss, security problems → *High priority*
* 🟠 **Normal** – a feature misbehaves under some circumstances → *Medium priority*
* 🟢 **Non-Critical** – cosmetic issues, typos, small requests → *Low priority*

It does **not** know your product or your users; a person should always make the final decision.
""")


# ---------------------------------------------------------------- Predict
def _init_form_state():
    for key in FORM_KEYS.values():
        st.session_state.setdefault(key, "")
    st.session_state.setdefault("example_choice", list(C.EXAMPLES)[0])


def load_example():
    ex = C.EXAMPLES[st.session_state["example_choice"]]
    for f, key in FORM_KEYS.items():
        st.session_state[key] = ex.get(f, "")
    st.session_state["pred_result"] = None
    st.session_state["loaded_example"] = st.session_state["example_choice"]


def clear_form():
    for key in FORM_KEYS.values():
        st.session_state[key] = ""
    st.session_state["pred_result"] = None
    st.session_state["loaded_example"] = None


def _examples_panel():
    with st.container(border=True):
        st.markdown("#### 💡 Try Example Bug Reports")
        st.caption("Not sure what to write? Load an example to fill in the form, then press **Predict Severity**.")
        st.selectbox("Choose an example", list(C.EXAMPLES), key="example_choice",
                     format_func=lambda n: f"{C.EXAMPLE_ICONS[n]}  {n}")
        ex = C.EXAMPLES[st.session_state["example_choice"]]
        st.markdown(f"> {ex['title']}")
        b = st.columns(2)
        b[0].button("Load Example", on_click=load_example, type="primary", width="stretch")
        b[1].button("Clear Form", on_click=clear_form, width="stretch")
        st.info("**Demonstration only.** These examples are illustrative inputs written for this app. They have "
                "no verified severity, and the model's answer for them is not a guaranteed result.", icon="ℹ️")
        st.caption("The model learned from Mozilla browser-engine reports, so its suggestions for web-app "
                   "style reports like these are especially uncertain.")
    with st.container(border=True):
        st.markdown("#### ✍️ Tips for a good report")
        st.markdown("- One problem per report\n- Say *what* is affected and *how badly*\n"
                    "- Mention crashes, data loss or security impact explicitly\n"
                    "- Paste the exact error text if you have one")


def _show_result(r: dict, meta: dict):
    st.markdown("### Prediction result")
    left, right = st.columns([2, 3], gap="large")
    with left:
        severity_result(r["severity"], r["priority"])
        if r["confidence"] is not None:
            st.metric(f"Model confidence ({confidence_label(meta)})", f"{r['confidence']:.0%}",
                      help="The model's probability for the predicted class. It is a score, not a guarantee "
                           "that the prediction is correct.")
    with right:
        if r["probabilities"]:
            pdf = pd.DataFrame({"Severity": CLASSES, "Probability": [r["probabilities"][c] for c in CLASSES]})
            fig = px.bar(pdf, x="Probability", y="Severity", orientation="h", color="Severity",
                         color_discrete_map=COLORS, text=pdf["Probability"].map("{:.0%}".format),
                         title="Class probability distribution")
            fig.update_layout(xaxis_range=[0, 1], xaxis_tickformat=".0%")
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(style_fig(fig, 250), width="stretch")
        else:
            st.info("This model does not provide probability estimates, so no confidence is shown.")

    if r["low_confidence"] is not False:
        st.warning(f"**{LOW_CONFIDENCE_MESSAGE}** "
                   + (f"The model's score for this class is {r['confidence']:.0%}, below the "
                      f"{r['threshold']:.0%} threshold." if r["confidence"] is not None
                      else "This model provides no probability estimate."), icon="🔎")
    st.warning(f"**Please review manually.** {EXPERIMENTAL_WARNING} "
               f"(Trained on {meta['n_total']} real Mozilla Bugzilla reports; Non-Critical bugs are the hardest "
               "for it to recognise.)", icon="⚠️")
    st.caption(r["disclaimer"])
    st.button("↺ Start a new report", on_click=clear_form, key="new_report")


def page_predict(pred):
    _init_form_state()
    hero("Predict Bug Severity",
         "Describe a single bug in plain language. Only the first two fields are required – the more you add, "
         "the more the model has to work with.")
    form_col, side_col = st.columns([3, 2], gap="large")
    with form_col:
        if st.session_state.get("loaded_example"):
            st.caption(f"📋 Loaded example: **{st.session_state['loaded_example']}** (demonstration input)")
        with st.form("predict", border=True):
            section("A", "Basic bug information", "required")
            title = st.text_input(f"{C.FIELD_LABELS['title']} *", key=FORM_KEYS["title"],
                                  placeholder=C.FIELD_PLACEHOLDERS["title"], help=C.FIELD_HELP["title"])
            st.caption(C.FIELD_HELP["title"])
            desc = st.text_area(f"{C.FIELD_LABELS['description']} *", key=FORM_KEYS["description"], height=120,
                                placeholder=C.FIELD_PLACEHOLDERS["description"], help=C.FIELD_HELP["description"])
            st.caption(C.FIELD_HELP["description"])

            section("B", "Reproduction details", "optional")
            steps_ = st.text_area(f"{C.FIELD_LABELS['steps']} (optional)", key=FORM_KEYS["steps"], height=130,
                                  placeholder=C.FIELD_PLACEHOLDERS["steps"], help=C.FIELD_HELP["steps"])
            st.caption(C.FIELD_HELP["steps"])
            c1, c2 = st.columns(2)
            expected = c1.text_area(f"{C.FIELD_LABELS['expected']} (optional)", key=FORM_KEYS["expected"],
                                    height=100, placeholder=C.FIELD_PLACEHOLDERS["expected"],
                                    help=C.FIELD_HELP["expected"])
            c1.caption(C.FIELD_HELP["expected"])
            actual = c2.text_area(f"{C.FIELD_LABELS['actual']} (optional)", key=FORM_KEYS["actual"], height=100,
                                  placeholder=C.FIELD_PLACEHOLDERS["actual"], help=C.FIELD_HELP["actual"])
            c2.caption(C.FIELD_HELP["actual"])

            section("C", "Technical details", "optional")
            error = st.text_area(f"{C.FIELD_LABELS['error']} (optional)", key=FORM_KEYS["error"], height=100,
                                 placeholder=C.FIELD_PLACEHOLDERS["error"], help=C.FIELD_HELP["error"])
            st.caption(C.FIELD_HELP["error"])
            submitted = st.form_submit_button("🔮 Predict Severity", type="primary", width="stretch")

        if submitted:
            values = {"title": title, "description": desc, "steps": steps_, "expected": expected,
                      "actual": actual, "error": error}
            missing = C.missing_required(values)
            st.session_state["pred_result"] = None
            if len(missing) == len(C.REQUIRED_FORM_FIELDS):
                st.error("The bug report is empty. Please fill in **What is the problem?** and "
                         "**Describe the bug** (or load an example).", icon="✋")
            elif missing:
                st.error(f"Please fill in the required field: **{missing[0]}**.", icon="✋")
            else:
                try:
                    st.session_state["pred_result"] = pred.predict_one(**values)
                except BugSeverityError as exc:
                    st.error(str(exc), icon="✋")
        if st.session_state.get("pred_result"):
            _show_result(st.session_state["pred_result"], pred.metadata)
    with side_col:
        _examples_panel()


# ---------------------------------------------------------------- Batch
def _csv_help():
    with st.expander("📘 How to Prepare Your CSV File"):
        st.markdown("""
1. Click **Download CSV Template** and open the file in Excel (or Google Sheets / LibreOffice).
2. Enter **one bug report per row**.
3. **Keep the column headers unchanged** (you may delete optional columns you don't use).
4. Save the file as **CSV** – in Excel: *File › Save As › CSV UTF-8 (Comma delimited)*.
5. Upload it below.
6. Check the **preview** and the validation summary.
7. Click **Predict All Bug Severities**.
8. Click **Download Prediction Results**.
""")
        st.markdown("**Example of a correctly filled file (illustrative):**")
        st.dataframe(C.template_dataframe(), hide_index=True, width="stretch")
    with st.expander("❓ Frequently asked questions"):
        st.markdown(f"""
**Can I upload an Excel file?**
Only CSV is supported. In Excel choose *File › Save As › CSV UTF-8 (Comma delimited)* and upload that file.
Both UTF-8 and Excel's regular "CSV (Comma delimited)" encodings are accepted.

**Can I upload multiple bugs?**
Yes – one bug per row. Files up to {C.MAX_ROWS:,} rows are processed in one go; split larger files.

**Do I need to enter severity labels?**
No. You only provide the bug text; the model generates the predicted severity. A `severity` column, if present,
is simply kept as an extra column and ignored by the prediction.

**Can I upload my own Bugzilla reports?**
Yes, as long as the file has at least one of the supported text columns (for Bugzilla exports, `summary`).

**Are predictions always correct?**
No. Predictions are experimental and require human verification.
""")


def page_batch(pred):
    hero("Batch Bug Severity Prediction",
         "Upload multiple bug reports in a CSV file and automatically predict their severity in one operation.")
    steps(["Download CSV Template", "Add Bug Reports", "Upload CSV", "Generate Predictions", "Download Results"])

    section("1", "Prepare your file", "CSV format, template and sample data")
    info_col, dl_col = st.columns([3, 2], gap="large")
    with info_col:
        st.markdown("Your CSV needs **at least one text column** from the table below – we recommend "
                    "`title` + `description`. Column names are not case-sensitive. **No severity column is "
                    "needed.** Any other columns (IDs, owners…) are kept unchanged in the results.")
        st.dataframe(pd.DataFrame(C.COLUMN_GUIDE), hide_index=True, width="stretch")
        st.caption("*Technically any one of the seven text columns is enough. A row with no text in any of "
                   "them is reported as invalid and skipped.")
    with dl_col:
        with st.container(border=True):
            st.markdown("#### 📥 Download CSV Template")
            st.caption(f"Correct headers ({', '.join(C.TEMPLATE_COLUMNS)}) plus 3 illustrative rows. "
                       "Replace the rows with your own bugs.")
            st.download_button("Download CSV Template", C.to_csv_bytes(C.template_dataframe()),
                               C.TEMPLATE_FILENAME, "text/csv", type="primary", width="stretch",
                               key="dl_template")
        with st.container(border=True):
            st.markdown("#### 🧪 Download Sample Bug Reports")
            st.caption("10 demonstration reports (login, crash, API, payment, link, layout, spelling, image, "
                       "search, validation). No severity labels – not real Bugzilla data, never used for training.")
            st.download_button("Download Sample Bug Reports", C.to_csv_bytes(C.sample_reports_dataframe()),
                               C.SAMPLE_FILENAME, "text/csv", width="stretch", key="dl_sample")
    _csv_help()

    section("2", "Upload Your Bug Reports", "Upload a CSV file containing one or more bug reports.")
    up = st.file_uploader("Upload a CSV file containing one or more bug reports.", type=["csv"],
                          key="batch_upload")
    if up is None:
        st.session_state.pop("batch_result", None)
        return
    data = up.getvalue()
    file_key = hashlib.sha1(data).hexdigest()
    try:
        df = C.read_upload(data)
    except BugSeverityError as exc:
        st.error(f"**This file could not be read.** {exc}", icon="🚫")
        return
    if df.empty:
        st.error("**The CSV has headers but no bug report rows.** Add at least one row and upload again.",
                 icon="🚫")
        return

    rep = C.inspect_upload(df)
    m = st.columns(4)
    m[0].metric("Records detected", rep["total"])
    m[1].metric("Valid records", rep["valid"])
    m[2].metric("Invalid records", rep["invalid"])
    m[3].metric("Missing required columns", "Yes" if rep["missing_required"] else "None")
    if rep["missing_required"]:
        st.error("**No supported text column found.** The file needs at least one of: "
                 + ", ".join(f"`{c}`" for c in C.TEMPLATE_COLUMNS[:2]) + " (or any of "
                 + ", ".join(f"`{r['Column']}`" for r in C.COLUMN_GUIDE) + "). "
                 f"Columns found: {', '.join(f'`{c}`' for c in rep['other_columns']) or 'none'}. "
                 "Tip: start from the CSV template.", icon="🚫")
    else:
        st.success(f"Text columns recognised: {', '.join(f'`{c}`' for c in rep['text_columns'])}"
                   + (f" · extra columns kept as-is: {', '.join(f'`{c}`' for c in rep['other_columns'])}"
                      if rep["other_columns"] else ""), icon="✅")
    if rep["invalid"]:
        rows = ", ".join(map(str, rep["invalid_rows"][:15])) + (" …" if len(rep["invalid_rows"]) > 15 else "")
        st.warning(f"{rep['invalid']} row(s) have no usable text and will be marked invalid "
                   f"(spreadsheet rows {rows}).", icon="⚠️")
    if rep["duplicates"]:
        rows = ", ".join(map(str, rep["duplicate_rows"][:15])) + (" …" if len(rep["duplicate_rows"]) > 15 else "")
        st.warning(f"{rep['duplicates']} row(s) repeat an earlier report's text (spreadsheet rows {rows}). "
                   "They will still be predicted – remove them if they are accidental.", icon="♻️")
    if rep["too_many_rows"]:
        st.error(f"The file has {rep['total']:,} rows; the limit is {C.MAX_ROWS:,}. Please split it.", icon="🚫")

    st.markdown("**Preview** (first 50 rows)")
    st.dataframe(df.head(50), width="stretch", hide_index=True)

    section("3", "Generate predictions")
    can_run = not rep["missing_required"] and rep["valid"] > 0 and not rep["too_many_rows"]
    if st.button("🔮 Predict All Bug Severities", type="primary", disabled=not can_run, key="run_batch"):
        with st.status(f"Predicting severity for {rep['valid']} report(s)…", expanded=False) as status:
            try:
                out = pred.predict_csv(io.BytesIO(data))
            except BugSeverityError as exc:
                status.update(label="Prediction failed", state="error")
                st.error(str(exc))
                return
            except Exception as exc:  # unexpected – show, don't hide
                status.update(label="Prediction failed", state="error")
                st.error(f"Unexpected error while processing the file: {type(exc).__name__}: {exc}")
                return
            status.update(label="Predictions complete", state="complete")
        st.session_state["batch_result"] = (file_key, C.restore_original_headers(out, df.columns))

    stored = st.session_state.get("batch_result")
    if not stored or stored[0] != file_key:
        if not can_run:
            st.caption("Fix the problems above to enable prediction.")
        return
    _batch_results(stored[1], pred.metadata)


def _batch_results(out: pd.DataFrame, meta: dict):
    section("4", "Results", "from the trained model – verify before acting")
    sev = out["predicted_severity"]
    counts = {c: int((sev == c).sum()) for c in CLASSES}
    invalid = int(sev.str.startswith("Invalid").sum())
    flagged = int((out["review"] == LOW_CONFIDENCE_MESSAGE).sum())
    m = st.columns(5)
    m[0].metric("Total processed", int(len(out) - invalid))
    for col, c in zip(m[1:4], CLASSES):
        col.metric(f"{ICONS[c]} {c}", counts[c])
    m[4].metric("🔎 Low confidence", flagged)
    if invalid:
        st.caption(f"{invalid} invalid row(s) (no usable text) were skipped.")

    view = pd.DataFrame({"Bug Report": C.report_preview(out.iloc[:, : len(out.columns) - 4]),
                         "Predicted Severity": sev, "Recommended Priority": out["priority"]})
    has_conf = not out["confidence"].isna().all()
    if has_conf:
        view["Confidence"] = out["confidence"]
    view["Review"] = out["review"]

    left, right = st.columns([3, 2], gap="large")
    with left:
        options = CLASSES + (["Invalid"] if invalid else [])
        chosen = st.pills("Filter by predicted severity", options, selection_mode="multi", default=options,
                          key="batch_filter",
                          format_func=lambda o: f"{ICONS.get(o, '⚪')} {o}")
        chosen = chosen or []
        mask = sev.isin([c for c in chosen if c != "Invalid"]) | (sev.str.startswith("Invalid") & ("Invalid" in chosen))
        cfg = {"Bug Report": st.column_config.TextColumn(width="large")}
        if has_conf:
            cfg["Confidence"] = st.column_config.ProgressColumn(
                f"Confidence ({confidence_label(meta)})", min_value=0.0, max_value=1.0, format="percent")
        st.dataframe(view[mask], width="stretch", hide_index=True, column_config=cfg)
        st.caption(f"Showing {int(mask.sum())} of {len(view)} rows.")
    with right:
        st.plotly_chart(bar(counts, "Predicted severity distribution", 320), width="stretch")

    st.download_button("📥 Download Prediction Results", C.results_csv(out), C.RESULTS_FILENAME, "text/csv",
                       type="primary", key="dl_results")
    st.caption("The download contains all your original columns plus `predicted_severity`, `priority`"
               + (", `confidence`" if has_conf else "") + " and `review`. Your uploaded file is not modified.")
    st.warning(f"**Please review manually.** {EXPERIMENTAL_WARNING}", icon="⚠️")


# ---------------------------------------------------------------- Analytics
def page_analytics(pred):
    meta = pred.metadata
    hero("Analytics Dashboard", "How the training data looks and how the current model performed on reports "
                                "it had not seen during training.")
    try:
        (orig_df, orig_info), ext = get_datasets()
    except BugSeverityError as exc:
        st.error(str(exc))
        return

    st.markdown("### 📦 Training data of the current model")
    if model_uses_collected_data(meta):
        st.caption("The current model was trained on the original sample plus the collected Bugzilla reports.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(bar(meta["class_counts"], "Severity distribution (mapped classes)"), width="stretch")
    with c2:
        raw = meta.get("raw_label_counts") or orig_info["raw_label_counts"]
        rdf = pd.DataFrame({"Original label": list(raw), "Reports": list(raw.values())})
        fig = px.bar(rdf, x="Original label", y="Reports", text="Reports",
                     title="Original Bugzilla severity labels", color_discrete_sequence=["#6366f1"])
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(style_fig(fig, 340), width="stretch")

    lens = orig_df["text"].str.split().str.len()
    s = st.columns(4)
    s[0].metric("Reports used", orig_info["rows_used"])
    s[1].metric("Duplicates removed", orig_info["dropped_duplicates"])
    s[2].metric("Mean words / report", f"{lens.mean():.1f}")
    s[3].metric("Max words", int(lens.max()))
    st.caption("Statistics of `data/bugzilla_summary_sample.csv` (the original sample).")

    st.markdown("### 🎯 Model evaluation")
    tm = meta["test_metrics"]
    sets = meta.get("test_sets", {meta.get("primary_test_set", "test"): tm})
    choice = list(sets)[0]
    if len(sets) > 1:
        choice = st.radio("Held-out test set", list(sets), horizontal=True,
                          help="ext_test: held-out collected Bugzilla reports; orig_test: the 20 original rows.")
    tm = sets[choice]
    st.caption(f"{tm['n_samples']} held-out reports. With so few rows (only "
               f"{tm['per_class']['Non-Critical']['support']} Non-Critical), one mistake moves a per-class "
               "score by many percentage points.")
    m = st.columns(4)
    m[0].metric("Accuracy", f"{tm['accuracy']:.2f}")
    m[1].metric("Macro precision", f"{tm['macro']['precision']:.2f}")
    m[2].metric("Macro recall", f"{tm['macro']['recall']:.2f}")
    m[3].metric("Macro F1", f"{tm['macro']['f1']:.2f}")
    b = meta.get("majority_baseline")
    if b:
        st.caption(f"For comparison, always guessing '{b['class']}' scores accuracy {b['accuracy']:.2f}, "
                   f"macro-F1 {b['macro_f1']:.2f}.")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("**Class-wise performance**")
        pc = pd.DataFrame(tm["per_class"]).T.rename_axis("Class").reset_index()
        st.dataframe(pc, hide_index=True, width="stretch", column_config={
            "precision": st.column_config.ProgressColumn("Precision", min_value=0, max_value=1, format="%.2f"),
            "recall": st.column_config.ProgressColumn("Recall", min_value=0, max_value=1, format="%.2f"),
            "f1": st.column_config.ProgressColumn("F1", min_value=0, max_value=1, format="%.2f"),
            "support": st.column_config.NumberColumn("Support", format="%d")})
        st.caption("Precision: how often a predicted class is right. Recall: how many real cases of a class "
                   "were found.")
    with right:
        cm = tm["confusion_matrix"]
        fig = go.Figure(go.Heatmap(z=cm, x=tm["labels"], y=tm["labels"], text=cm, texttemplate="%{text}",
                                   colorscale="Blues", showscale=False))
        fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual",
                          yaxis_autorange="reversed")
        st.plotly_chart(style_fig(fig, 360), width="stretch")

    if "risk_coverage" in tm:
        st.markdown("### 🔎 Confidence vs. accuracy")
        rc = pd.DataFrame(tm["risk_coverage"])
        st.dataframe(rc.rename(columns={"threshold": "Threshold", "coverage": "Share answered",
                                        "accuracy_confident": "Accuracy (answered)",
                                        "accuracy_flagged": "Accuracy (flagged)", "n_flagged": "Flagged"}),
                     hide_index=True, width="stretch")

    if ext is not None and not model_uses_collected_data(meta):
        ext_df, ext_info = ext
        st.markdown("### 🗂️ Additional collected data (not used by the current model)")
        st.caption(f"{len(ext_df):,} further real Mozilla Bugzilla reports (2006–2008) collected from the public "
                   "Bugzilla API for research (see `data/README.md`). The deployed model has **not** been "
                   "trained or evaluated on them.")
        st.plotly_chart(bar({c: int((ext_df["label"] == c).sum()) for c in CLASSES},
                            "Collected reports – severity distribution", 300), width="stretch")


# ---------------------------------------------------------------- Model information
def page_model(pred):
    meta, tm = pred.metadata, pred.metadata["test_metrics"]
    hero("Model Information", "What the model is, what it was trained on, how it was evaluated and where it "
                              "falls short.")
    k = st.columns(4)
    with k[0]:
        kpi("Algorithm", meta["algorithm"], "selected by cross-validated macro-F1")
    with k[1]:
        kpi("Training reports", meta["n_total"], f"{meta['n_train']} train · {meta['n_test']} held-out test")
    with k[2]:
        kpi("Held-out accuracy", f"{tm['accuracy']:.0%}", f"n = {tm['n_samples']}")
    with k[3]:
        kpi("Macro-F1", f"{tm['macro']['f1']:.2f}", "average over the 3 classes")

    st.write("")
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.markdown("#### ⚙️ How it works")
            st.markdown(f"""
* **Feature extraction:** {meta['feature_extraction']}
* **Classes:** {', '.join(meta['classes'])}
* **Probabilities:** {confidence_label(meta)}; reports below a {meta.get('confidence_threshold', 0.6):.0%}
  score are flagged *{LOW_CONFIDENCE_MESSAGE}*
* **Trained (UTC):** {meta['trained_at_utc']} · seed {meta['seed']}
* **Versions:** {', '.join(f'{k} {v}' for k, v in meta['versions'].items())}
""")
    with b:
        with st.container(border=True):
            st.markdown("#### 📚 Training data")
            st.markdown(f"""
* Real Mozilla Bugzilla bug summaries, **{meta['n_total']} reports**
* Classes: {', '.join(f'{ICONS[c]} {c} {meta["class_counts"][c]}' for c in CLASSES)}
* Stratified split: {meta['n_train']} for training, {meta['n_test']} held out for testing;
  the final model is then re-fitted on all reports
* Label mapping: critical/blocker/major → Critical · normal → Normal · minor/trivial/enhancement → Non-Critical
""")

    st.markdown("#### 🏁 Model comparison (cross-validation on the training split)")
    cols = {"cv_macro_f1_mean": "CV macro-F1", "cv_macro_f1_std": "± std", "cv_accuracy_mean": "CV accuracy",
            "cv_weighted_f1_mean": "CV weighted-F1", "cv_noncritical_recall_mean": "CV Non-Critical recall"}
    comp = pd.DataFrame(meta["model_comparison"]).T
    comp = comp[[c for c in cols if c in comp.columns]].astype(float).rename(columns=cols)
    comp = comp.sort_values("CV macro-F1", ascending=False).rename_axis("Model").reset_index()
    comp["Model"] = [f"{m}  ✅" if m.split(" | ")[0] == meta["algorithm"] and i == 0 else m
                     for i, m in enumerate(comp["Model"])]
    st.dataframe(comp, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in comp.columns[1:]})
    st.caption("Differences of a few hundredths are within the noise of such a small dataset.")

    prev = meta.get("previous_result")
    if prev and "orig_test" in meta.get("test_sets", {}):
        o = meta["test_sets"]["orig_test"]
        st.markdown("#### Previous vs. current (same 20 original test rows)")
        st.dataframe(pd.DataFrame({
            "Previous (RF, 79 rows)": [prev["accuracy"], prev["macro_f1"], prev["noncritical_recall"]],
            "Current": [o["accuracy"], o["macro"]["f1"], o["per_class"]["Non-Critical"]["recall"]]},
            index=["Accuracy", "Macro-F1", "Non-Critical recall"]), width="stretch")

    st.markdown("#### ⚠️ Known limitations")
    nc = tm["per_class"]["Non-Critical"]
    st.warning(f"""
* {'Data' if model_uses_collected_data(meta) else 'Small dataset'} ({meta['n_total']} reports) from one project and era
  (Mozilla, {'2006–2008' if model_uses_collected_data(meta) else '2007'}) – metrics {'' if model_uses_collected_data(meta) else '**have very high variance** and '}are not evidence of real-world accuracy elsewhere.
* Non-Critical is the rarest class ({meta['class_counts']['Non-Critical']} reports); on the held-out test the model
  found {round(nc['recall'] * nc['support'])} of {nc['support']} Non-Critical reports.
* Bugzilla labels are subjective; mapping `major→Critical` and `enhancement→Non-Critical` is a documented judgement call.
* Reports written very differently from short Mozilla bug summaries (e.g. modern web-app tickets) are outside what the
  model has seen.
* Confidence is a model score, not a guarantee for an individual report.
* Not validated for production use – a person must confirm every severity.
""")


# ---------------------------------------------------------------- About
def page_about():
    hero("About the Project", "An honest, lightweight machine-learning assistant for bug triage.")
    a, b = st.columns(2, gap="large")
    with a:
        card("Problem statement", "Developers spend a lot of time manually reading and sorting incoming bug "
             "reports. Critical issues can wait in the queue behind trivial ones.", "🧩")
    with b:
        card("Objectives", "Classify a bug report as Critical / Normal / Non-Critical, map it to a High / Medium "
             "/ Low priority, support single and batch use, and report evaluation honestly.", "🎯")
    st.write("")
    st.markdown("### 🛠️ Technologies")
    tech = ["Python 3.14", "pandas", "NumPy", "scikit-learn", "SciPy", "Plotly", "Streamlit", "joblib", "pytest"]
    st.markdown(" ".join(f"`{t}`" for t in tech))

    st.markdown("### 🏗️ Architecture & workflow")
    st.graphviz_chart("""digraph {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#6366f1", fontname="Helvetica", fontsize=11];
      edge [color="#6366f1"];
      "Bug report\\n(form or CSV)" -> "Validation" -> "Text cleaning\\n(keeps error codes)" -> "TF-IDF\\nfeatures"
        -> "Classifier" -> "Severity" -> "Priority" -> "Human review";
    }""")
    st.markdown("""
| Layer | Files |
|---|---|
| User interface | `app.py`, `ui/theme.py`, `ui/content.py` |
| Prediction API | `src/predict.py` (`SeverityPredictor`) |
| Data & NLP | `src/preprocessing.py`, `src/data.py` |
| Training & evaluation | `src/train.py`, `src/modeling.py`, `src/evaluation.py`, `src/experiments.py` |
""")
    st.markdown("### 🚀 Future improvements")
    st.markdown("- Validation on other projects and issue trackers\n- Using full bug descriptions, not only "
                "summaries\n- Transformer-based text embeddings\n- Learning from triagers' corrections\n"
                "- Component / product prediction")
    st.caption("Based on [ozsaygin/severity-prediction](https://github.com/ozsaygin/severity-prediction) "
               "(Word2Vec + MLP, Siemens Turkey / Sabancı University, MIT licence); modernised – see "
               "`PROJECT_AUDIT.md`.")


# ---------------------------------------------------------------- navigation
PAGES = {"🏠 Home": page_home, "🔮 Predict": page_predict, "📄 Batch Prediction": page_batch,
         "📊 Analytics": page_analytics, "🧠 Model Info": page_model, "ℹ️ About": page_about}

sidebar_brand()
choice = st.sidebar.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed")
with st.sidebar:
    st.divider()
    try:
        _m = get_predictor().metadata
        st.markdown(f"**Model:** {_m['algorithm']}  \n<small>trained on {_m['n_total']} real Bugzilla "
                    "reports</small>", unsafe_allow_html=True)
    except (ModelNotFoundError, BugSeverityError):
        st.markdown("**Model:** not loaded")
    st.markdown("  \n".join(f"{ICONS[k]} {k} → **{v}**" for k, v in PRIORITY.items()))
    st.caption("⚠️ Predictions are experimental recommendations. Always verify manually.")

if choice == "ℹ️ About":
    page_about()
else:
    PAGES[choice](load_or_stop())
