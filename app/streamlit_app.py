import json
import os
from pathlib import Path

import requests
import streamlit as st

API_BASE = os.environ.get("MEDISIGHT_API_URL", "http://localhost:8000")
REPORTS_DIR = Path(__file__).parent.parent / "reports"

st.set_page_config(
    page_title="MediSight AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("MediSight AI")
st.sidebar.caption("Clinical Decision Support")
page = st.sidebar.radio(
    "Navigate",
    ["Patient Risk Score", "Clinical Q&A", "Model Metrics"],
    index=0,
)

try:
    r = requests.get(f"{API_BASE}/health", timeout=3)
    api_ok = r.status_code == 200
except Exception:
    api_ok = False

status = "API: connected" if api_ok else "API: offline — start with `uvicorn api.main:app`"
st.sidebar.markdown(f"{'🟢' if api_ok else '🔴'} {status}")


def page_risk_score():
    st.title("30-Day Readmission Risk Score")
    st.caption(
        "Fill in the patient's discharge-time features. "
        "The model scores at discharge, so all fields reflect the completed encounter."
    )

    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            age = st.number_input("Age at admission", 0, 120, 65)
            gender = st.selectbox("Gender", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0])
            insurance = st.selectbox(
                "Insurance risk tier",
                [(1, "Low risk (private)"), (2, "Medium (Medicare/Medicaid)"), (3, "High (uninsured)")],
                format_func=lambda x: x[1],
            )

        with col2:
            st.subheader("This Encounter")
            enc_class = st.selectbox(
                "Encounter class",
                [(1, "Inpatient"), (2, "Emergency"), (0, "Other")],
                format_func=lambda x: x[1],
            )
            los = st.number_input("Length of stay (days)", 0.0, 365.0, 3.0, step=0.5)
            num_dx = st.number_input("# diagnoses this visit", 0, 50, 3)
            num_labs = st.number_input("# labs this visit", 0, 200, 10)
            num_abnormal = st.number_input("# abnormal labs", 0, 200, 2)
            avg_dev = st.number_input("Avg lab deviation (z-score)", 0.0, 10.0, 0.5, step=0.1)
            num_meds = st.number_input("# medications this visit", 0, 50, 5)

        with col3:
            st.subheader("History & Comorbidities")
            prior_6m  = st.number_input("Prior admissions (6 months)", 0, 20, 0)
            prior_12m = st.number_input("Prior admissions (12 months)", 0, 20, 0)
            prior_tot = st.number_input("Prior admissions (total)", 0, 100, 1)
            days_prev = st.number_input(
                "Days since previous visit (blank = first admission)",
                min_value=0, max_value=3650, value=0,
                help="Leave as 0 and check 'First admission' if this is the patient's first visit.",
            )
            is_first = st.checkbox("First admission", value=(prior_tot == 0))
            comorbidities = st.number_input("Comorbidity count (prior)", 0, 30, 1)

            st.markdown("**Chronic conditions**")
            has_hf  = st.checkbox("Heart failure")
            has_dm  = st.checkbox("Diabetes")
            has_copd = st.checkbox("COPD")
            has_ckd  = st.checkbox("CKD")
            has_htn  = st.checkbox("Hypertension")

        submitted = st.form_submit_button("Score Patient", type="primary")

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

        label = result["risk_label"]
        score = result["risk_score"]
        los_pred = result["predicted_los_days"]

        color = {"high": "red", "medium": "orange", "low": "green"}.get(label, "gray")
        badge = f":{color}[**{label.upper()} RISK**]"

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Readmission Probability", f"{score:.1%}")
        c2.metric("Risk Level", label.upper())
        c3.metric("Predicted LOS (days)", f"{los_pred:.1f}")

        st.markdown(f"### {badge}")

        if result.get("shap_values"):
            st.subheader("Top feature contributions (SHAP)")
            shap_vals = result["shap_values"]
            sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            names = [s[0] for s in sorted_shap]
            vals  = [s[1] for s in sorted_shap]

            import altair as alt
            import pandas as pd
            shap_df = pd.DataFrame({"feature": names, "shap_value": vals})
            shap_df["color"] = shap_df["shap_value"].apply(lambda v: "positive" if v > 0 else "negative")
            chart = (
                alt.Chart(shap_df)
                .mark_bar()
                .encode(
                    x=alt.X("shap_value:Q", title="SHAP value (impact on risk)"),
                    y=alt.Y("feature:N", sort="-x", title="Feature"),
                    color=alt.Color(
                        "color:N",
                        scale=alt.Scale(domain=["positive", "negative"], range=["#e74c3c", "#2ecc71"]),
                        legend=None,
                    ),
                    tooltip=["feature", alt.Tooltip("shap_value:Q", format=".4f")],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.subheader("Or look up a patient by ID")
    with st.form("patient_lookup"):
        pid = st.text_input("Patient ID (from database)", placeholder="e.g. abc123...")
        lookup = st.form_submit_button("Load Patient")

    if lookup and pid.strip():
        with st.spinner("Fetching..."):
            try:
                resp = requests.get(f"{API_BASE}/patient/{pid.strip()}/summary", timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.HTTPError as exc:
                if exc.response.status_code == 404:
                    st.warning(f"Patient {pid} not found.")
                else:
                    st.error(f"API error: {exc}")
                return
            except requests.RequestException as exc:
                st.error(f"API error: {exc}")
                return

        demo = data.get("demographics", {})
        enc  = data.get("most_recent_encounter")
        risk = data.get("risk_score")

        st.markdown(
            f"**{demo.get('first_name', '?')} {demo.get('last_name', '?')}** — "
            f"{demo.get('gender', '?')}, born {demo.get('birth_date', '?')}"
        )

        if enc:
            st.markdown(
                f"Most recent: **{enc.get('encounter_class', '?')}** admission "
                f"{enc.get('admission_date', '')[:10]} — {enc.get('primary_diagnosis', '?')}, "
                f"LOS {enc.get('length_of_stay_days', '?')} days"
            )
        if risk:
            label = risk["risk_label"]
            color = {"high": "red", "medium": "orange", "low": "green"}.get(label, "gray")
            st.markdown(
                f"Readmission risk: :{color}[**{label.upper()}**] "
                f"({risk['readmission_probability']:.1%}) — "
                f"Predicted LOS: {risk['predicted_los_days']} days"
            )


def page_clinical_qa():
    st.title("Clinical Q&A")
    st.caption(
        "Ask a natural-language question about a patient's clinical history. "
        "Answers are grounded in clinical notes with source citations."
    )

    pid = st.text_input(
        "Patient ID (optional — leave blank to search all patients)",
        placeholder="e.g. 259d5017-...",
    )
    question = st.text_area(
        "Clinical question",
        placeholder="What medications was this patient on? "
                    "What were the main diagnoses? "
                    "Summarize the discharge plan.",
        height=100,
    )

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Searching clinical notes..."):
            try:
                payload = {
                    "question": question.strip(),
                    "patient_id": pid.strip() if pid.strip() else None,
                }
                resp = requests.post(f"{API_BASE}/ask", json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                st.error(f"API error: {exc}")
                return

        st.subheader("Answer")
        st.markdown(data["answer"])

        if data.get("sources"):
            with st.expander(f"Sources ({len(data['sources'])} notes)"):
                for i, src in enumerate(data["sources"], 1):
                    st.markdown(
                        f"**[{i}] {src.get('note_type', '?')}** — "
                        f"{src.get('note_date', '?')} | "
                        f"Patient: `{src.get('patient_id', '?')}`"
                    )
                    st.caption(src.get("excerpt", "")[:400] + "…")
                    st.divider()


def page_model_metrics():
    st.title("Model Performance & Explainability")

    tab1, tab2 = st.tabs(["Readmission Classifier", "LOS Regressor"])

    with tab1:
        st.subheader("XGBoost Readmission Classifier")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ROC AUC",  "0.9005")
            st.metric("PR AUC",   "0.6996")
            st.metric("F1 Score", "0.6127")
        with col2:
            st.metric("Recall",    "0.83", help="At threshold 0.68")
            st.metric("Precision", "0.49", help="At threshold 0.68")
            st.metric("Threshold", "0.68")

        st.markdown(
            "The optimal threshold (0.68) was chosen to maximise recall — "
            "false negatives (missed readmissions) carry a higher clinical cost "
            "than false positives (unnecessary follow-up)."
        )

        shap_summary = REPORTS_DIR / "shap_summary.png"
        shap_waterfall = REPORTS_DIR / "shap_waterfall_example.png"

        if shap_summary.exists():
            st.subheader("SHAP Feature Importance (global)")
            st.image(str(shap_summary), use_container_width=True)
        else:
            st.info("Run `python -m models.train` to generate SHAP plots.")

        if shap_waterfall.exists():
            st.subheader("SHAP Waterfall (example patient)")
            st.image(str(shap_waterfall), use_container_width=True)

    with tab2:
        st.subheader("GBM Length-of-Stay Regressor")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MAE",  "0.663 days")
            st.metric("RMSE", "2.227 days", help="RMSE higher than MAE indicates long-tail outliers (stays > 14 days)")
            st.metric("R²",   "0.643")
        with col2:
            st.markdown(
                "**Training details**\n\n"
                "- Target: `log1p(length_of_stay_days)` — log-transforms LOS to "
                "compress long-tail outliers before fitting\n"
                "- Predictions are inverse-transformed with `expm1` at inference time\n"
                "- Trained on inpatient encounters only"
            )

        dq_report = REPORTS_DIR / "data_quality_report.html"
        if dq_report.exists():
            st.subheader("Data Quality Report")
            st.markdown(
                f"[Open full report]({dq_report}) — generated by ydata-profiling on `ml_encounter_features`",
                unsafe_allow_html=True,
            )


if page == "Patient Risk Score":
    page_risk_score()
elif page == "Clinical Q&A":
    page_clinical_qa()
elif page == "Model Metrics":
    page_model_metrics()
