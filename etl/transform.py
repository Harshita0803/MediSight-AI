"""
Transforms raw extracted records into clean, analytics-ready DataFrames
and computes ML feature columns.
"""

import logging
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INSURANCE_RISK_TIERS = {
    "Medicare": 3,
    "Medicaid": 4,
    "Dual Eligible": 4,
    "NO_INSURANCE": 4,
    "unknown": 4,
}
DEFAULT_INSURANCE_TIER = 2  # private/commercial


def _parse_date(val: str | None) -> date | None:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return None


def _age_on(birth_date: date | None, reference_date: date) -> int | None:
    if not birth_date:
        return None
    years = reference_date.year - birth_date.year
    if (reference_date.month, reference_date.day) < (birth_date.month, birth_date.day):
        years -= 1
    return max(years, 0)


def _age_bucket(age: int | None) -> str:
    if age is None:
        return "unknown"
    if age <= 17:
        return "pediatric"
    if age <= 40:
        return "adult"
    if age <= 65:
        return "middle"
    return "senior"


def build_patients_df(raw_patients: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_patients)
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce").dt.date
    df["gender"] = df["gender"].str.lower().fillna("unknown")

    # Normalise insurance names for risk tier mapping
    df["insurance_type"] = (
        df["insurance_type"]
        .fillna("unknown")
        .str.strip()
    )

    # Deduplicate on patient_id (keep first)
    df = df.drop_duplicates(subset="patient_id").reset_index(drop=True)
    logger.info("Patients: %d rows", len(df))
    return df


def build_encounters_df(raw_encounters: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_encounters)
    df["admission_date"] = pd.to_datetime(df["admission_date"], utc=True, errors="coerce")
    df["discharge_date"] = pd.to_datetime(df["discharge_date"], utc=True, errors="coerce")

    df["length_of_stay_days"] = (
        (df["discharge_date"] - df["admission_date"])
        .dt.total_seconds()
        .div(86_400)
        .round(2)
    )
    # Negative or implausible LOS → null
    df.loc[df["length_of_stay_days"] < 0, "length_of_stay_days"] = np.nan
    df.loc[df["length_of_stay_days"] > 365, "length_of_stay_days"] = np.nan

    df["total_claim_cost"] = pd.to_numeric(df["total_claim_cost"], errors="coerce")
    df["payer_coverage"] = pd.to_numeric(df["payer_coverage"], errors="coerce")

    df = df.drop_duplicates(subset="encounter_id").reset_index(drop=True)
    logger.info("Encounters: %d rows", len(df))
    return df


def build_diagnoses_df(raw_diagnoses: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_diagnoses)
    df["onset_date"] = pd.to_datetime(df["onset_date"], errors="coerce").dt.date
    df["abatement_date"] = pd.to_datetime(df["abatement_date"], errors="coerce").dt.date
    df["icd_code"] = df["icd_code"].fillna("UNKNOWN")
    df = df.drop_duplicates(subset="diagnosis_id").reset_index(drop=True)
    logger.info("Diagnoses: %d rows", len(df))
    return df


def build_labs_df(raw_labs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_labs)
    df["observation_date"] = pd.to_datetime(df["observation_date"], utc=True, errors="coerce")
    df["value_numeric"] = pd.to_numeric(df["value_numeric"], errors="coerce")
    df["reference_low"] = pd.to_numeric(df["reference_low"], errors="coerce")
    df["reference_high"] = pd.to_numeric(df["reference_high"], errors="coerce")
    df = df.drop_duplicates(subset="lab_id").reset_index(drop=True)
    logger.info("Labs: %d rows", len(df))
    return df


def build_medications_df(raw_medications: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_medications)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce").dt.date
    df = df.drop_duplicates(subset="medication_id").reset_index(drop=True)
    logger.info("Medications: %d rows", len(df))
    return df


# ─── 30-day readmission label ─────────────────────────────────────────────────

def add_readmission_label(df_enc: pd.DataFrame) -> pd.DataFrame:
    """
    For each inpatient/emergency encounter, set readmitted_30d=True if the same
    patient has another encounter starting within 30 days of this discharge.
    """
    df = df_enc.copy()
    df = df.sort_values(["patient_id", "admission_date"])

    inpatient_mask = df["encounter_class"].isin(["IMP", "EMER", "inpatient", "emergency"])
    df_ip = df[inpatient_mask].copy()

    readmitted = {}
    for pid, group in df_ip.groupby("patient_id"):
        group = group.sort_values("admission_date").reset_index()
        for i in range(len(group) - 1):
            discharge = group.at[i, "discharge_date"]
            next_admit = group.at[i + 1, "admission_date"]
            if pd.notna(discharge) and pd.notna(next_admit):
                days_gap = (next_admit - discharge).total_seconds() / 86_400
                if 0 < days_gap <= 30:
                    readmitted[group.at[i, "encounter_id"]] = True

    df["readmitted_30d"] = df["encounter_id"].map(readmitted).fillna(False)
    logger.info(
        "Readmission labels: %d positive out of %d inpatient encounters",
        df["readmitted_30d"].sum(),
        inpatient_mask.sum(),
    )
    return df


# ─── ML feature engineering ───────────────────────────────────────────────────

CHRONIC_ICD_PREFIXES = (
    "E10", "E11",   # diabetes
    "I10",          # hypertension
    "J44",          # COPD
    "I50",          # heart failure
    "N18",          # CKD
    "F32", "F33",   # depression
    "J45",          # asthma
    "E78",          # hyperlipidemia
    "M79", "M54",   # musculoskeletal pain
)


def build_ml_features(
    df_patients: pd.DataFrame,
    df_encounters: pd.DataFrame,
    df_diagnoses: pd.DataFrame,
    df_labs: pd.DataFrame,
    df_medications: pd.DataFrame,
) -> pd.DataFrame:
    today = date.today()

    # ── patient base ──────────────────────────────────────────────────────────
    feat = df_patients[["patient_id", "birth_date", "gender", "insurance_type"]].copy()
    feat["age"] = feat["birth_date"].apply(lambda d: _age_on(d, today))
    feat["age_bucket"] = feat["age"].apply(_age_bucket)
    feat["gender_encoded"] = (feat["gender"] == "male").astype(int)
    feat["insurance_risk_tier"] = feat["insurance_type"].apply(
        lambda x: INSURANCE_RISK_TIERS.get(x, DEFAULT_INSURANCE_TIER)
    )

    # ── encounter aggregates ──────────────────────────────────────────────────
    enc_agg = (
        df_encounters.groupby("patient_id")
        .agg(
            total_encounters=("encounter_id", "count"),
            last_los_days=("length_of_stay_days", "last"),
        )
        .reset_index()
    )
    feat = feat.merge(enc_agg, on="patient_id", how="left")

    # Prior inpatient admissions in last 12 months
    cutoff_12m = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
    df_ip = df_encounters[df_encounters["encounter_class"].isin(["IMP", "EMER", "inpatient", "emergency"])]
    prior_12m = (
        df_ip[df_ip["admission_date"] >= cutoff_12m]
        .groupby("patient_id")["encounter_id"]
        .count()
        .rename("prior_admissions_12m")
        .reset_index()
    )
    prior_total = (
        df_ip.groupby("patient_id")["encounter_id"]
        .count()
        .rename("prior_admissions_total")
        .reset_index()
    )
    feat = feat.merge(prior_12m, on="patient_id", how="left")
    feat = feat.merge(prior_total, on="patient_id", how="left")

    # Days since last visit
    last_visit = (
        df_encounters.groupby("patient_id")["admission_date"]
        .max()
        .rename("last_visit_date")
        .reset_index()
    )
    now_utc = pd.Timestamp.now(tz="UTC")
    feat = feat.merge(last_visit, on="patient_id", how="left")
    feat["days_since_last_visit"] = (
        (now_utc - feat["last_visit_date"]).dt.total_seconds() / 86_400
    ).round(0).astype("Int64")

    # ── diagnosis aggregates ──────────────────────────────────────────────────
    diag_count = (
        df_diagnoses.groupby("patient_id")["icd_code"]
        .nunique()
        .rename("comorbidity_count")
        .reset_index()
    )
    chronic = df_diagnoses[
        df_diagnoses["icd_code"].str[:3].isin([p[:3] for p in CHRONIC_ICD_PREFIXES])
    ]
    chronic_count = (
        chronic.groupby("patient_id")["icd_code"]
        .nunique()
        .rename("chronic_condition_count")
        .reset_index()
    )
    feat = feat.merge(diag_count, on="patient_id", how="left")
    feat = feat.merge(chronic_count, on="patient_id", how="left")

    # ── lab deviation (mean z-score vs reference range) ───────────────────────
    labs_with_ref = df_labs.dropna(subset=["value_numeric", "reference_low", "reference_high"])
    labs_with_ref = labs_with_ref[labs_with_ref["reference_high"] > labs_with_ref["reference_low"]].copy()
    mid = (labs_with_ref["reference_low"] + labs_with_ref["reference_high"]) / 2
    rang = (labs_with_ref["reference_high"] - labs_with_ref["reference_low"]) / 2
    labs_with_ref["z_score"] = (labs_with_ref["value_numeric"] - mid) / rang.replace(0, np.nan)
    lab_dev = (
        labs_with_ref.groupby("patient_id")["z_score"]
        .apply(lambda s: s.abs().mean())
        .rename("avg_lab_deviation")
        .reset_index()
    )
    feat = feat.merge(lab_dev, on="patient_id", how="left")

    # ── medication count ──────────────────────────────────────────────────────
    med_count = (
        df_medications.groupby("patient_id")["medication_id"]
        .count()
        .rename("medication_count")
        .reset_index()
    )
    feat = feat.merge(med_count, on="patient_id", how="left")

    # ── fill nulls ────────────────────────────────────────────────────────────
    int_cols = [
        "total_encounters", "prior_admissions_12m", "prior_admissions_total",
        "comorbidity_count", "chronic_condition_count", "medication_count",
    ]
    for col in int_cols:
        if col in feat.columns:
            feat[col] = feat[col].fillna(0).astype(int)

    feat["avg_lab_deviation"] = feat["avg_lab_deviation"].fillna(0.0)
    feat["last_los_days"] = feat["last_los_days"].fillna(0.0)

    feat = feat.drop(columns=["birth_date", "gender", "insurance_type", "last_visit_date"], errors="ignore")
    logger.info("ML features: %d patients, %d columns", len(feat), len(feat.columns))
    return feat
