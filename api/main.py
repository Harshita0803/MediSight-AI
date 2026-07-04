import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
import sqlalchemy
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel

from models.train import ALL_FEATURES, NUMERIC_FEATURES, BOOLEAN_FEATURES

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODELS_DIR = Path(__file__).parent.parent / "models"
FAISS_DIR = Path(__file__).parent.parent / "data" / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
READMISSION_THRESHOLD = 0.68

ML_STATE: dict = {}


def _db_engine():
    db_url = os.environ["DATABASE_URL"].replace("postgres://", "postgresql://")
    if "sslmode" not in db_url:
        db_url += "?sslmode=require"
    return sqlalchemy.create_engine(db_url)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models...")
    clf_pipeline = joblib.load(MODELS_DIR / "readmission_classifier.pkl")
    los_pipeline = joblib.load(MODELS_DIR / "los_regressor.pkl")

    preprocessor = clf_pipeline[:-1]
    clf_step = clf_pipeline["clf"]
    explainer = shap.TreeExplainer(clf_step)

    logger.info("Loading FAISS index...")
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        str(FAISS_DIR), embeddings_model, allow_dangerous_deserialization=True
    )

    ML_STATE["clf_pipeline"] = clf_pipeline
    ML_STATE["los_pipeline"] = los_pipeline
    ML_STATE["preprocessor"] = preprocessor
    ML_STATE["explainer"] = explainer
    ML_STATE["vectorstore"] = vectorstore
    ML_STATE["embeddings"] = embeddings_model

    logger.info("All models loaded — API ready.")
    yield
    ML_STATE.clear()


app = FastAPI(
    title="MediSight AI",
    description="Clinical decision support: readmission risk scoring and RAG Q&A",
    version="0.1.0",
    lifespan=lifespan,
)


class EncounterFeatures(BaseModel):
    age_at_admission: float
    gender_encoded: int
    insurance_risk_tier: int
    encounter_class_encoded: int
    length_of_stay_days: float
    num_diagnoses_this_visit: int
    num_labs_this_visit: int
    num_abnormal_labs_this_visit: int
    avg_lab_deviation_this_visit: float
    num_meds_this_visit: int
    prior_admissions_6m: int
    prior_admissions_12m: int
    prior_admissions_total: int
    days_since_previous_visit: Optional[float] = None
    comorbidity_count_prior: int
    has_heart_failure: bool
    has_diabetes: bool
    has_copd: bool
    has_ckd: bool
    has_hypertension: bool
    is_first_admission: int


class RiskScoreResponse(BaseModel):
    risk_score: float
    risk_label: str
    threshold_used: float
    predicted_los_days: float
    shap_values: dict[str, float]


class AskRequest(BaseModel):
    question: str
    patient_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


def _features_to_df(features: EncounterFeatures) -> pd.DataFrame:
    row = features.model_dump()
    return pd.DataFrame([row])[ALL_FEATURES]


def _risk_label(score: float, threshold: float) -> str:
    if score >= threshold:
        return "high"
    if score >= threshold * 0.6:
        return "medium"
    return "low"


def _compute_shap(row_df: pd.DataFrame) -> dict[str, float]:
    preprocessor = ML_STATE["preprocessor"]
    explainer = ML_STATE["explainer"]
    X_transformed = preprocessor.transform(row_df)
    sv = explainer.shap_values(X_transformed)
    try:
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [n.split("__")[-1] for n in feature_names]
    except Exception:
        feature_names = ALL_FEATURES
    values = sv[0] if hasattr(sv, "__len__") and not isinstance(sv[0], float) else sv
    return dict(zip(feature_names, [float(v) for v in values]))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/risk-score", response_model=RiskScoreResponse)
def risk_score(features: EncounterFeatures):
    row_df = _features_to_df(features)

    clf = ML_STATE["clf_pipeline"]
    los = ML_STATE["los_pipeline"]

    prob = float(clf.predict_proba(row_df)[0][1])
    predicted_los = float(np.expm1(los.predict(row_df)[0]))
    shap_dict = _compute_shap(row_df)

    return RiskScoreResponse(
        risk_score=round(prob, 4),
        risk_label=_risk_label(prob, READMISSION_THRESHOLD),
        threshold_used=READMISSION_THRESHOLD,
        predicted_los_days=round(max(predicted_los, 0.0), 2),
        shap_values=shap_dict,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    from rag.chain import build_rag_chain
    from rag.retriever import _FAISSRetriever

    retriever = _FAISSRetriever(
        vectorstore=ML_STATE["vectorstore"],
        embeddings_model=ML_STATE["embeddings"],
        patient_id=request.patient_id,
    )
    chain = build_rag_chain(
        patient_id=request.patient_id,
        vectorstore=ML_STATE["vectorstore"],
        embeddings_model=ML_STATE["embeddings"],
    )
    answer = chain.invoke(request.question)

    docs = retriever.invoke(request.question)
    sources = [
        {
            "note_type":    doc.metadata.get("note_type", "?"),
            "note_date":    doc.metadata.get("note_date", "?"),
            "patient_id":   doc.metadata.get("patient_id", "?"),
            "admission_id": doc.metadata.get("admission_id", "?"),
            "excerpt":      doc.page_content[:300],
        }
        for doc in docs
    ]

    return AskResponse(answer=answer, sources=sources)


@app.get("/patient/{patient_id}/summary")
def patient_summary(patient_id: str):
    try:
        engine = _db_engine()
        with engine.connect() as conn:
            patient = pd.read_sql(
                "SELECT * FROM dim_patient WHERE patient_id = %(pid)s",
                conn, params={"pid": patient_id},
            )
            encounter_features = pd.read_sql(
                """
                SELECT ef.*, fe.admission_date, fe.discharge_date,
                       fe.primary_diagnosis_desc, fe.encounter_class
                FROM ml_encounter_features ef
                JOIN fact_encounters fe USING (encounter_id)
                WHERE ef.patient_id = %(pid)s
                ORDER BY fe.admission_date DESC
                LIMIT 1
                """,
                conn, params={"pid": patient_id},
            )
        engine.dispose()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if patient.empty:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    demographics = patient.iloc[0].where(patient.iloc[0].notna(), other=None).to_dict()

    if encounter_features.empty:
        return {"demographics": demographics, "most_recent_encounter": None, "risk_score": None}

    enc = encounter_features.iloc[0]

    feature_cols = [c for c in ALL_FEATURES if c in enc.index]
    row_df = enc[feature_cols].to_frame().T.reset_index(drop=True)
    for col in ALL_FEATURES:
        if col not in row_df.columns:
            row_df[col] = np.nan
    row_df = row_df[ALL_FEATURES]

    clf = ML_STATE["clf_pipeline"]
    los = ML_STATE["los_pipeline"]
    prob = float(clf.predict_proba(row_df)[0][1])
    predicted_los = float(np.expm1(los.predict(row_df)[0]))

    encounter_info = {
        "encounter_id":        enc.get("encounter_id"),
        "admission_date":      str(enc.get("admission_date", "")),
        "discharge_date":      str(enc.get("discharge_date", "")),
        "encounter_class":     enc.get("encounter_class"),
        "primary_diagnosis":   enc.get("primary_diagnosis_desc"),
        "length_of_stay_days": enc.get("length_of_stay_days"),
    }

    return {
        "demographics": demographics,
        "most_recent_encounter": encounter_info,
        "risk_score": {
            "readmission_probability": round(prob, 4),
            "risk_label":              _risk_label(prob, READMISSION_THRESHOLD),
            "predicted_los_days":      round(max(predicted_los, 0.0), 2),
        },
    }
