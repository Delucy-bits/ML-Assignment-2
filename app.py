"""
app.py — Streamlit app for the Breast Cancer classification assignment.

Visual design: a "diagnostic report" theme (custom fonts/colors/layout via
injected CSS) rather than default Streamlit styling. The confusion matrix and
classification report are built as plain HTML/CSS instead of a matplotlib
plot, so there's no headless-backend dependency and the styling matches the
rest of the page exactly.

Features (per assignment Step 6):
  a. Dataset upload option (CSV)          -> st.file_uploader
  b. Model selection dropdown             -> st.selectbox
  c. Display of evaluation metrics        -> headline + secondary stat cards
  d. Confusion matrix / classification report -> custom HTML grid + table
"""

import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Breast Cancer Classifier — Diagnostic Report", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl",
}
CLASS_LABELS = ("Malignant", "Benign")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --bg: #F6F7F5;
  --surface: #FFFFFF;
  --ink: #1C2321;
  --ink-soft: #5B655F;
  --line: #DBDFD9;
  --accent: #2E6E62;
  --accent-soft: #E4EFEC;
  --alert: #A8442F;
  --alert-soft: #F4E4E0;
}

.stApp { background: var(--bg); }
.block-container { max-width: 980px; padding-top: 2.5rem; padding-bottom: 4rem; }
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.report-eyebrow, .section-eyebrow, .headline-label, .secondary-label {
  font-family: 'IBM Plex Mono', monospace;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-soft);
}
.report-eyebrow { font-size: 12px; color: var(--accent); font-weight: 500; margin-bottom: 8px; }
.report-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 34px; font-weight: 600; color: var(--ink);
  margin: 0 0 10px 0; letter-spacing: -0.01em;
}
.report-subtitle { font-size: 15px; color: var(--ink-soft); max-width: 640px; line-height: 1.6; }
.report-divider { border: none; border-top: 1px solid var(--line); margin: 26px 0; }

.section-eyebrow { font-size: 11px; margin: 0 0 4px 0; }
.section-title { font-size: 20px; font-weight: 600; color: var(--ink); margin: 0 0 18px 0; }

.headline-row { display: flex; gap: 16px; margin-bottom: 16px; }
.headline-stat { flex: 1; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 20px 24px; }
.headline-label { font-size: 11px; margin-bottom: 6px; }
.headline-value { font-family: 'IBM Plex Mono', monospace; font-size: 40px; font-weight: 600; color: var(--accent); line-height: 1; }

.secondary-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }
.secondary-stat { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; }
.secondary-label { font-size: 10px; margin-bottom: 6px; }
.secondary-value { font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 600; color: var(--ink); }

.cm-wrap { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 20px; height: 100%; }
.cm-grid { display: grid; grid-template-columns: 96px 1fr 1fr; gap: 4px; }
.cm-colhead, .cm-rowhead { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--ink-soft); display: flex; align-items: center; justify-content: center; text-align: center; padding: 6px; }
.cm-rowhead { justify-content: flex-end; text-align: right; padding-right: 10px; }
.cm-cell { font-family: 'IBM Plex Mono', monospace; font-size: 25px; font-weight: 600; display: flex; align-items: center; justify-content: center; border-radius: 4px; min-height: 72px; }
.cm-cell.correct { background: var(--accent-soft); color: var(--accent); }
.cm-cell.wrong { background: var(--alert-soft); color: var(--alert); }

.report-panel { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 20px; height: 100%; }
.report-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; }
.report-table th { text-align: right; font-weight: 500; color: var(--ink-soft); font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.05em; padding: 6px 8px; border-bottom: 1px solid var(--line); }
.report-table th:first-child, .report-table td:first-child { text-align: left; }
.report-table td { text-align: right; padding: 8px; color: var(--ink); border-bottom: 1px solid var(--line); }
.report-table tr.summary td { color: var(--ink-soft); font-style: italic; }
.report-table tr:last-child td { border-bottom: none; }
.report-table td.best { background: var(--accent-soft); color: var(--accent); font-weight: 600; }

.empty-state { background: var(--surface); border: 1px dashed var(--line); border-radius: 6px; padding: 30px; text-align: center; color: var(--ink-soft); font-size: 14px; margin-top: 8px; }

[data-testid="stFileUploader"] section { background: var(--surface); border: 1px dashed var(--line); border-radius: 6px; }
[data-testid="stFileUploader"] label p, [data-testid="stSelectbox"] label p {
  font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--ink-soft);
}
</style>
"""


@st.cache_resource
def load_pickle(path: str):
    return joblib.load(path)


METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_model_results(display_name: str, X_scaled, y_true) -> dict:
    """Run one model on the uploaded data and return its predictions + all 6 metrics."""
    model = load_pickle(MODEL_FILES[display_name])
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def render_overall_comparison(all_results: dict) -> str:
    """All 5 models x all 6 metrics, computed on the SAME uploaded data. Best value per column is highlighted."""
    best = {m: max(r[m] for r in all_results.values()) for m in METRIC_NAMES}
    header = "".join(f"<th>{m}</th>" for m in METRIC_NAMES)
    body = ""
    for name, r in all_results.items():
        cells = ""
        for m in METRIC_NAMES:
            cls = ' class="best"' if abs(r[m] - best[m]) < 1e-9 else ""
            cells += f"<td{cls}>{r[m]:.3f}</td>"
        body += f"<tr><td>{name}</td>{cells}</tr>"
    return f"""<div class="report-panel"><table class="report-table">
      <thead><tr><th></th>{header}</tr></thead>
      <tbody>{body}</tbody></table></div>"""


def render_confusion_grid(cm) -> str:
    tl, tr = int(cm[0][0]), int(cm[0][1])
    bl, br = int(cm[1][0]), int(cm[1][1])
    m, b = CLASS_LABELS
    return f"""
    <div class="cm-wrap">
      <div class="cm-grid">
        <div></div>
        <div class="cm-colhead">Pred. {m.lower()}</div>
        <div class="cm-colhead">Pred. {b.lower()}</div>
        <div class="cm-rowhead">Actual {m.lower()}</div>
        <div class="cm-cell correct">{tl}</div>
        <div class="cm-cell wrong">{tr}</div>
        <div class="cm-rowhead">Actual {b.lower()}</div>
        <div class="cm-cell wrong">{bl}</div>
        <div class="cm-cell correct">{br}</div>
      </div>
    </div>"""


def render_classification_table(report: dict) -> str:
    m, b = CLASS_LABELS
    rows = ""
    for label in (m, b):
        r = report[label]
        rows += (f"<tr><td>{label}</td><td>{r['precision']:.3f}</td>"
                 f"<td>{r['recall']:.3f}</td><td>{r['f1-score']:.3f}</td>"
                 f"<td>{int(r['support'])}</td></tr>")
    macro, weighted = report["macro avg"], report["weighted avg"]
    total = int(macro["support"])
    rows += f'<tr class="summary"><td>Accuracy</td><td colspan="3">{report["accuracy"]:.3f}</td><td>{total}</td></tr>'
    rows += (f"<tr><td>Macro avg</td><td>{macro['precision']:.3f}</td>"
             f"<td>{macro['recall']:.3f}</td><td>{macro['f1-score']:.3f}</td><td>{total}</td></tr>")
    rows += (f"<tr><td>Weighted avg</td><td>{weighted['precision']:.3f}</td>"
             f"<td>{weighted['recall']:.3f}</td><td>{weighted['f1-score']:.3f}</td><td>{total}</td></tr>")
    return f"""<div class="report-panel"><table class="report-table">
      <thead><tr><th></th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
      <tbody>{rows}</tbody></table></div>"""



def get_model_ranking(all_results: dict) -> pd.DataFrame:
    """Rank models using F1 as the primary overall performance measure."""
    rows = []
    for name, result in all_results.items():
        rows.append({
            "Model": name,
            "Accuracy": result["Accuracy"],
            "AUC": result["AUC"],
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "F1": result["F1"],
            "MCC": result["MCC"],
        })

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        by=["F1", "Accuracy", "MCC"],
        ascending=False,
    ).reset_index(drop=True)
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    return ranking


def model_interpretation(model_name: str, metrics: dict) -> str:
    return (
        f"{model_name} achieved an accuracy of {metrics['Accuracy']:.3f}, "
        f"with precision of {metrics['Precision']:.3f}, recall of "
        f"{metrics['Recall']:.3f}, and an F1 score of {metrics['F1']:.3f}. "
        f"Its ROC-AUC was {metrics['AUC']:.3f} and MCC was "
        f"{metrics['MCC']:.3f}, indicating strong classification performance "
        "on the uploaded test set."
    )


st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="report-eyebrow">Diagnostic report &middot; Breast Cancer Wisconsin dataset</div>
<div class="report-title">Classifier comparison</div>
<div class="report-subtitle">Five classical models trained on the same 569-patient dataset,
evaluated on six metrics. Upload the test set below and inspect any model's full report.</div>
<hr class="report-divider">
""", unsafe_allow_html=True)

st.markdown('<div class="section-eyebrow">Run a model</div><div class="section-title">Upload test data</div>', unsafe_allow_html=True)

col_upload, col_select = st.columns([2, 1])
with col_upload:
    uploaded_file = st.file_uploader("Test data (CSV)", type=["csv"], label_visibility="visible")
with col_select:
    model_choice = st.selectbox("Model", list(MODEL_FILES.keys()))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("The uploaded CSV must include a 'target' column with the true labels "
                 "(0 = malignant, 1 = benign). Use the provided test_data.csv.")
    else:
        X = df.drop(columns=["target"])
        y_true = df["target"]

        st.markdown(
            f"""
            <div class="status-card">
                ✓ <strong>{uploaded_file.name}</strong> loaded
                &nbsp;&middot;&nbsp; {len(df)} test samples
                &nbsp;&middot;&nbsp; {X.shape[1]} features
                &nbsp;&middot;&nbsp; Target column detected
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            scaler = load_pickle("model/scaler.pkl")
            X_scaled = scaler.transform(X)

            all_results = {name: compute_model_results(name, X_scaled, y_true) for name in MODEL_FILES}

            selected = all_results[model_choice]

            st.markdown(f'<div class="section-eyebrow">Report</div><div class="section-title">{model_choice}</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="headline-row">
              <div class="headline-stat"><div class="headline-label">Accuracy</div><div class="headline-value">{selected['Accuracy']:.3f}</div></div>
              <div class="headline-stat"><div class="headline-label">AUC</div><div class="headline-value">{selected['AUC']:.3f}</div></div>
            </div>
            <div class="secondary-row">
              <div class="secondary-stat"><div class="secondary-label">Precision</div><div class="secondary-value">{selected['Precision']:.3f}</div></div>
              <div class="secondary-stat"><div class="secondary-label">Recall</div><div class="secondary-value">{selected['Recall']:.3f}</div></div>
              <div class="secondary-stat"><div class="secondary-label">F1 score</div><div class="secondary-value">{selected['F1']:.3f}</div></div>
              <div class="secondary-stat"><div class="secondary-label">MCC</div><div class="secondary-value">{selected['MCC']:.3f}</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="dataset-card">
                    <div class="dataset-title">Interpretation</div>
                    <div class="dataset-subtitle">
                        {model_interpretation(model_choice, selected)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            cm = confusion_matrix(y_true, selected["y_pred"])
            report = classification_report(y_true, selected["y_pred"], target_names=list(CLASS_LABELS), output_dict=True)

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown('<div class="section-eyebrow">Confusion matrix</div>', unsafe_allow_html=True)
                st.markdown(render_confusion_grid(cm), unsafe_allow_html=True)
            with col_right:
                st.markdown('<div class="section-eyebrow">Classification report</div>', unsafe_allow_html=True)
                st.markdown(render_classification_table(report), unsafe_allow_html=True)

            # Overall comparison is intentionally the FINAL section.
            st.markdown('<hr class="report-divider">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-eyebrow">Final summary</div>'
                '<div class="section-title">Overall model comparison</div>',
                unsafe_allow_html=True,
            )

            ranking = get_model_ranking(all_results)
            best = ranking.iloc[0]

            st.markdown(
                f"""
                <div class="headline-row">
                  <div class="headline-stat">
                    <div class="headline-label">Best overall model</div>
                    <div class="headline-value" style="font-size: 28px;">
                        {best['Model']}
                    </div>
                  </div>
                  <div class="headline-stat">
                    <div class="headline-label">Best F1 score</div>
                    <div class="headline-value">{best['F1']:.3f}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Compact ranked table.
            ranked_display = ranking.copy()
            ranked_display["Accuracy"] = ranked_display["Accuracy"].map(lambda x: f"{x:.3f}")
            ranked_display["AUC"] = ranked_display["AUC"].map(lambda x: f"{x:.3f}")
            ranked_display["Precision"] = ranked_display["Precision"].map(lambda x: f"{x:.3f}")
            ranked_display["Recall"] = ranked_display["Recall"].map(lambda x: f"{x:.3f}")
            ranked_display["F1"] = ranked_display["F1"].map(lambda x: f"{x:.3f}")
            ranked_display["MCC"] = ranked_display["MCC"].map(lambda x: f"{x:.3f}")

            st.dataframe(
                ranked_display,
                use_container_width=True,
                hide_index=True,
            )

            # Simple visual comparison.
            chart_df = ranking.set_index("Model")[["Accuracy", "F1"]]
            st.markdown(
                '<div class="section-eyebrow">Performance overview</div>',
                unsafe_allow_html=True,
            )
            st.bar_chart(chart_df)

            st.caption(
                "Models are ranked primarily by F1 score, with Accuracy and MCC "
                "used as tie-breakers. Higher values are better for all metrics."
            )

        except ValueError as e:
            st.error("Couldn't run predictions on this file — check that it has the same "
                     f"30 feature columns as test_data.csv. Details: {e}")
else:
    st.markdown('<div class="empty-state">Upload test_data.csv (included in this repo) to generate a report. '
                'Only test data should be uploaded here, per the assignment\'s Streamlit free-tier constraint.</div>',
                unsafe_allow_html=True)
