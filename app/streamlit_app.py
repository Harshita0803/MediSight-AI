import os
from pathlib import Path

import requests
import streamlit as st

API_BASE = os.environ.get("MEDISIGHT_API_URL", "http://localhost:8000")
REPORTS_DIR = Path(__file__).parent.parent / "reports"

st.set_page_config(
    page_title="MediSight AI",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
[data-testid="stAppViewContainer"] { background: #f5f7fb; }
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
h1 { color: #0f172a; font-weight: 700; letter-spacing: -0.5px; }
h2, h3 { color: #1e293b; font-weight: 600; }
.stTabs [role="tablist"] { gap: 4px; }
.stTabs [role="tab"] { font-weight: 500; }
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    background: #ffffff;
}
.risk-card {
    border-radius: 10px;
    padding: 20px 24px;
    margin: 12px 0;
    font-size: 15px;
    font-weight: 600;
    border-left: 4px solid;
}
.risk-high  { background:#fef2f2; border-color:#ef4444; color:#b91c1c; }
.risk-medium{ background:#fffbeb; border-color:#f59e0b; color:#b45309; }
.risk-low   { background:#f0fdf4; border-color:#22c55e; color:#15803d; }
.info-block {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px;
    margin: 8px 0;
}
.source-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 6px 0;
    font-size: 13px;
}
[data-testid="baseButton-primary"] {
    background-color: #3b82f6 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


def _api_status() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


with st.sidebar:
    st.markdown("## ⚕ MediSight AI")
    st.caption("Clinical Decision Support System")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Risk Scoring", "Clinical Q&A", "Model Performance"],
        label_visibility="collapsed",
    )

    st.divider()
    api_ok = _api_status()
    dot = "🟢" if api_ok else "🔴"
    conn_label = "API connected" if api_ok else "API offline"
    st.markdown(f"{dot} {conn_label}")
    if not api_ok:
        st.caption("`uvicorn api.main:app --reload`")

    st.divider()
    st.caption(
        "Trained on Synthea synthetic EHR data.  \n"
        "For research and demonstration purposes only."
    )


def page_risk_score():
    st.title("30-Day Readmission Risk Scoring")
    st.markdown(
        "Enter discharge-time patient data to generate a readmission risk score, "
        "predicted length of stay, and SHAP feature attributions."
    )
    st.divider()

    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            age = st.number_input("Age at admission", 0, 120, 65)
            gender = st.selectbox(
                "Gender",
                [("Female", 0), ("Male", 1)],
                format_func=lambda x: x[0],
            )
            insurance = st.selectbox(
                "Insurance tier",
                [(1, "Private / Commercial"), (2, "Medicare / Medicaid"), (3, "Uninsured")],
                format_func=lambda x: x[1],
            )

        with col2:
            st.markdown("**Current Encounter**")
            enc_class = st.selectbox(
                "Encounter type",
                [(1, "Inpatient"), (2, "Emergency"), (0, "Other")],
                format_func=lambda x: x[1],
            )
            los = st.number_input("Length of stay (days)", 0.0, 365.0, 3.0, step=0.5)
            num_dx = st.number_input("Diagnoses this visit", 0, 50, 3)
            num_labs = st.number_input("Lab tests ordered", 0, 200, 10)
            num_abnormal = st.number_input("Abnormal lab results", 0, 200, 2)
            avg_dev = st.number_input("Avg lab deviation (z-score)", 0.0, 10.0, 0.5, step=0.1)
            num_meds = st.number_input("Medications prescribed", 0, 50, 5)

        with col3:
            st.markdown("**History & Comorbidities**")
            prior_6m = st.number_input("Prior admissions — last 6 months", 0, 20, 0)
            prior_12m = st.number_input("Prior admissions — last 12 months", 0, 20, 0)
            prior_tot = st.number_input("Prior admissions — lifetime", 0, 100, 1)
            days_prev = st.number_input(
                "Days since prior admission",
                min_value=0, max_value=3650, value=0,
            )
            is_first = st.checkbox("First admission", value=(prior_tot == 0))
            comorbidities = st.number_input("Prior comorbidity count", 0, 30, 1)
            st.markdown("**Active chronic conditions**")
            has_hf = st.checkbox("Heart failure")
            has_dm = st.checkbox("Diabetes")
            has_copd = st.checkbox("COPD")
            has_ckd = st.checkbox("CKD")
            has_htn = st.checkbox("Hypertension")

        submitted = st.form_submit_button(
            "Generate Risk Score", type="primary", use_container_width=True
        )

    if submitted:
        payload = {
            "age_at_admission": age,
            "gender_encoded": gender[1],
            "insurance_risk_tier": insurance[0],
            "encounter_class_encoded": enc_class[0],
            "length_of_stay_days": los,
            "num_diagnoses_this_visit": num_dx,
            "num_labs_this_visit": num_labs,
            "num_abnormal_labs_this_visit": num_abnormal,
            "avg_lab_deviation_this_visit": avg_dev,
            "num_meds_this_visit": num_meds,
            "prior_admissions_6m": prior_6m,
            "prior_admissions_12m": prior_12m,
            "prior_admissions_total": prior_tot,
            "days_since_previous_visit": None if is_first else float(days_prev),
            "comorbidity_count_prior": comorbidities,
            "has_heart_failure": has_hf,
            "has_diabetes": has_dm,
            "has_copd": has_copd,
            "has_ckd": has_ckd,
            "has_hypertension": has_htn,
            "is_first_admission": int(is_first),
        }

        with st.spinner("Scoring..."):
            try:
                resp = requests.post(f"{API_BASE}/risk-score", json=payload, timeout=30)
                resp.raise_for_status()
                result = resp.json()
            except requests.RequestException as exc:
                st.error(f"API error: {exc}")
                return

        risk_lbl = result["risk_label"]
        score = result["risk_score"]
        los_pred = result["predicted_los_days"]

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Readmission Probability", f"{score:.1%}")
        c2.metric("Predicted LOS", f"{los_pred:.1f} days")
        c3.metric("Decision Threshold", "0.68")

        risk_labels = {"high": "HIGH RISK", "medium": "MODERATE RISK", "low": "LOW RISK"}
        st.markdown(
            f'<div class="risk-card risk-{risk_lbl}">'
            f'{risk_labels.get(risk_lbl, risk_lbl.upper())}</div>',
            unsafe_allow_html=True,
        )

        if result.get("shap_values"):
            st.subheader("Feature Attributions (SHAP)")
            st.caption(
                "Positive values increase predicted readmission risk; "
                "negative values decrease it."
            )
            shap_vals = result["shap_values"]
            sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

            import altair as alt
            import pandas as pd

            shap_df = pd.DataFrame(
                {
                    "Feature": [s[0].replace("_", " ") for s in sorted_shap],
                    "SHAP value": [s[1] for s in sorted_shap],
                }
            )
            shap_df["direction"] = shap_df["SHAP value"].apply(
                lambda v: "Increases risk" if v > 0 else "Decreases risk"
            )
            chart = (
                alt.Chart(shap_df)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("SHAP value:Q", title="SHAP value"),
                    y=alt.Y("Feature:N", sort="-x", title=None),
                    color=alt.Color(
                        "direction:N",
                        scale=alt.Scale(
                            domain=["Increases risk", "Decreases risk"],
                            range=["#ef4444", "#22c55e"],
                        ),
                        legend=alt.Legend(title=None, orient="bottom"),
                    ),
                    tooltip=[
                        alt.Tooltip("Feature:N"),
                        alt.Tooltip("SHAP value:Q", format=".4f"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.subheader("Patient Lookup")
    with st.form("patient_lookup"):
        pid = st.text_input("Patient ID", placeholder="e.g. abc123...")
        lookup = st.form_submit_button("Load Patient")

    if lookup and pid.strip():
        with st.spinner("Fetching patient record..."):
            try:
                resp = requests.get(f"{API_BASE}/patient/{pid.strip()}/summary", timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.HTTPError as exc:
                if exc.response.status_code == 404:
                    st.warning(f"Patient `{pid}` not found.")
                else:
                    st.error(f"API error: {exc}")
                return
            except requests.RequestException as exc:
                st.error(f"API error: {exc}")
                return

        demo = data.get("demographics", {})
        enc = data.get("most_recent_encounter")
        risk = data.get("risk_score")

        st.markdown(
            f'<div class="info-block">'
            f'<strong>{demo.get("first_name", "?")} {demo.get("last_name", "?")}</strong>'
            f'&nbsp;·&nbsp;{demo.get("gender", "?").title()}'
            f'&nbsp;·&nbsp;DOB {demo.get("birth_date", "?")}',
            unsafe_allow_html=True,
        )
        if enc:
            st.markdown(
                f'<p style="margin:8px 0 0 0;color:#64748b;font-size:14px;">'
                f'Most recent: <strong>{enc.get("encounter_class", "?")}</strong> — '
                f'{enc.get("primary_diagnosis", "?")} &nbsp;·&nbsp; '
                f'Admitted {str(enc.get("admission_date", ""))[:10]} &nbsp;·&nbsp; '
                f'LOS {enc.get("length_of_stay_days", "?")} days</p>',
                unsafe_allow_html=True,
            )
        if risk:
            rlbl = risk["risk_label"]
            st.markdown(
                f'<p style="margin:10px 0 0 0;font-size:14px;">'
                f'Readmission risk: <strong>{rlbl.upper()}</strong> '
                f'({risk["readmission_probability"]:.1%}) &nbsp;·&nbsp; '
                f'Predicted LOS: {risk["predicted_los_days"]} days</p></div>',
                unsafe_allow_html=True,
            )


def page_clinical_qa():
    st.title("Clinical Q&A")
    st.markdown(
        "Ask natural-language questions about a patient's clinical history. "
        "The system retrieves the most relevant note excerpts and answers from them, "
        "with source citations."
    )

    with st.expander("Usage guidance", expanded=False):
        st.markdown(
            """
**Supported questions** (per-patient narrative):
- What medications was this patient on at discharge?
- What diagnoses were documented during this admission?
- Summarize the discharge plan.
- What lab findings were noted?

**Not supported** (aggregate across all patients):
- How many patients have diabetes?
- List all patients with COPD.

Provide a Patient ID to focus the search on a single patient's notes.
"""
        )

    pid = st.text_input(
        "Patient ID (recommended)",
        placeholder="Leave blank to search across all patients",
    )
    question = st.text_area(
        "Clinical question",
        placeholder="What medications was this patient on at discharge?",
        height=90,
    )

    if st.button("Submit", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        aggregate_words = ("how many", "count", "number of patients", "list all patients")
        if any(w in question.lower() for w in aggregate_words) and not pid.strip():
            st.info(
                "This looks like an aggregate question. The Q&A searches note excerpts "
                "and cannot count across the full dataset. Try a per-patient question, "
                "or provide a Patient ID."
            )
            return

        with st.spinner("Searching clinical notes..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/ask",
                    json={"question": question.strip(), "patient_id": pid.strip() or None},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                st.error(f"API error: {exc}")
                return

        st.divider()
        st.subheader("Answer")
        st.markdown(data["answer"])

        if data.get("sources"):
            with st.expander(f"Source notes ({len(data['sources'])} retrieved)"):
                for i, src in enumerate(data["sources"], 1):
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>[{i}] {src.get("note_type", "?")} &nbsp;·&nbsp; '
                        f'{src.get("note_date", "?")} &nbsp;·&nbsp; '
                        f'Patient <code>{src.get("patient_id", "?")}</code></strong><br/>'
                        f'<span style="color:#64748b">{src.get("excerpt", "")[:350]}…</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


def page_model_performance():
    st.title("Model Performance")
    st.markdown(
        "Performance metrics, SHAP feature attributions, and evaluation results "
        "for both models trained on Synthea synthetic EHR data."
    )
    st.divider()

    tab1, tab2, tab3 = st.tabs(
        ["Readmission Classifier", "Length-of-Stay Regressor", "NL Faithfulness Study"]
    )

    with tab1:
        st.subheader("XGBoost 30-Day Readmission Classifier")
        st.caption("Discharge-time model trained with patient-level train/test split (80/20, stratified).")

        c1, c2, c3 = st.columns(3)
        c1.metric("ROC AUC", "0.9005")
        c2.metric("PR AUC", "0.6996")
        c3.metric("F1 Score", "0.6127")

        c4, c5, c6 = st.columns(3)
        c4.metric("Recall", "0.83", help="At optimal threshold 0.68")
        c5.metric("Precision", "0.49", help="At optimal threshold 0.68")
        c6.metric("Optimal Threshold", "0.68")

        st.divider()
        st.markdown(
            "The classification threshold (0.68) was selected via out-of-fold F1 maximisation "
            "on the training set. A recall-weighted operating point reflects the asymmetric "
            "cost of missed readmissions relative to false alerts in a clinical context."
        )

        shap_summary = REPORTS_DIR / "shap_summary.png"
        shap_waterfall = REPORTS_DIR / "shap_waterfall_example.png"

        if shap_summary.exists():
            st.subheader("Global Feature Importance (SHAP)")
            st.image(str(shap_summary), use_container_width=True)
        else:
            st.info("Run `python -m models.train` to generate SHAP visualisations.")

        if shap_waterfall.exists():
            st.subheader("Per-Patient Attribution (SHAP Waterfall)")
            st.image(str(shap_waterfall), use_container_width=True)

    with tab2:
        st.subheader("GBM Length-of-Stay Regressor")
        st.caption("Trained on inpatient encounters; target log-transformed to compress long-tail stays.")

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", "0.663 days")
        c2.metric("R²", "0.643")
        c3.metric("RMSE", "2.227 days")

        st.divider()
        st.markdown(
            "**Target transformation** — `length_of_stay_days` is log-transformed before fitting "
            "(`log1p`) to reduce the influence of extreme stays (>14 days) on the squared-error "
            "objective. Predictions are inverse-transformed at inference time (`expm1`)."
        )
        st.markdown(
            "The MAE/RMSE gap reflects residual variance from rare prolonged stays that "
            "are genuinely difficult to predict from discharge-time features alone."
        )

    with tab3:
        st.subheader("Natural-Language Faithfulness Evaluation")
        st.markdown(
            "Measures whether LLM-generated explanations faithfully preserve the SHAP attributions "
            "they are derived from. Two conditions were evaluated on a cohort of 200 encounters."
        )

        st.markdown("#### Condition A — Grounded (control)")
        st.caption(
            "Model given the SHAP top-5 features with signed contributions and asked to restate them. "
            "Establishes the sanity ceiling for the harness."
        )
        cA1, cA2, cA3, cA4 = st.columns(4)
        cA1.metric("Coverage", "0.633")
        cA2.metric("Fabrication", "0.000")
        cA3.metric("Direction Accuracy", "0.737")
        cA4.metric("Rank Fidelity", "0.910")

        st.markdown("#### Condition B — Reasoning (experimental)")
        st.caption(
            "Model given raw feature values only (SHAP signs withheld) and asked to reason "
            "about readmission drivers."
        )
        cB1, cB2, cB3, cB4 = st.columns(4)
        cB1.metric("Coverage", "0.306", delta="-0.327", delta_color="inverse")
        cB2.metric("Fabrication", "0.585", delta="+0.585", delta_color="inverse")
        cB3.metric("Direction Accuracy", "0.853", delta="+0.116")
        cB4.metric("Rank Fidelity", "-0.125", delta="-1.035", delta_color="inverse")

        st.divider()
        st.markdown("**Cross-method consistency (SHAP vs LIME) — Condition A**")
        lA1, lA2, lA3 = st.columns(3)
        lA1.metric("Jaccard Overlap", "0.419")
        lA2.metric("Sign Agreement", "0.907")
        lA3.metric("Rank Correlation (τ)", "0.506")

        st.divider()
        st.markdown(
            "**Parser validation** — mention-detection Cohen's κ = 0.835 (target ≥ 0.80 ✓). "
            "Direction κ = 0.627 (raw agreement 0.844; imbalance-adjusted). "
            "Evaluated against 90 hand-labelled explanation–feature pairs."
        )


if page == "Risk Scoring":
    page_risk_score()
elif page == "Clinical Q&A":
    page_clinical_qa()
elif page == "Model Performance":
    page_model_performance()
