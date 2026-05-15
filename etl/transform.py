"""
Transforms raw extracted records into clean, analytics-ready DataFrames
and computes ML feature columns.
"""

import logging
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as _skl_tts

from etl.ml_config import INPATIENT_CLASSES, SPLIT_RANDOM_STATE, SPLIT_TEST_SIZE

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


def build_patients_df(raw_patients: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(raw_patients)
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce").dt.date
    df["gender"] = df["gender"].str.lower().fillna("unknown")
    df["insurance_type"] = df["insurance_type"].fillna("unknown").str.strip()
    df = df.drop_duplicates(subset="patient_id").reset_index(drop=True)
    logger.info("Patients: %d rows", len(df))
    return df


def build_encounters_df(raw_encounters: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(raw_encounters)
    df["admission_date"] = pd.to_datetime(df["admission_date"], utc=True, errors="coerce")
    df["discharge_date"] = pd.to_datetime(df["discharge_date"], utc=True, errors="coerce")

    df["length_of_stay_days"] = (
        (df["discharge_date"] - df["admission_date"])
        .dt.total_seconds()
        .div(86_400)
        .round(2)
    )
    df.loc[df["length_of_stay_days"] < 0, "length_of_stay_days"] = np.nan
    df.loc[df["length_of_stay_days"] > 365, "length_of_stay_days"] = np.nan

    df["total_claim_cost"] = pd.to_numeric(df["total_claim_cost"], errors="coerce")
    df["payer_coverage"] = pd.to_numeric(df["payer_coverage"], errors="coerce")

    df = df.drop_duplicates(subset="encounter_id").reset_index(drop=True)
    logger.info("Encounters: %d rows", len(df))
    return df


def build_diagnoses_df(raw_diagnoses: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(raw_diagnoses)
    df["onset_date"] = pd.to_datetime(df["onset_date"], errors="coerce").dt.date
    df["abatement_date"] = pd.to_datetime(df["abatement_date"], errors="coerce").dt.date
    df["icd_code"] = df["icd_code"].fillna("UNKNOWN")
    df = df.drop_duplicates(subset="diagnosis_id").reset_index(drop=True)
    logger.info("Diagnoses: %d rows", len(df))
    return df


def build_labs_df(raw_labs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(raw_labs)
    df["observation_date"] = pd.to_datetime(df["observation_date"], utc=True, errors="coerce")
    df["value_numeric"] = pd.to_numeric(df["value_numeric"], errors="coerce")
    df["reference_low"] = pd.to_numeric(df["reference_low"], errors="coerce")
    df["reference_high"] = pd.to_numeric(df["reference_high"], errors="coerce")
    df = df.drop_duplicates(subset="lab_id").reset_index(drop=True)
    logger.info("Labs: %d rows", len(df))
    return df


def build_medications_df(raw_medications: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(raw_medications)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce").dt.date
    df = df.drop_duplicates(subset="medication_id").reset_index(drop=True)
    logger.info("Medications: %d rows", len(df))
    return df


# ─── 30-day readmission label ─────────────────────────────────────────────────

def add_readmission_label(df_enc: pd.DataFrame) -> pd.DataFrame:
    """
    For each inpatient/emergency encounter, set readmitted_30d=True if the same
    patient has another inpatient admission within 30 days of this discharge.

    Censoring: encounters whose discharge_date falls within 30 days of the dataset's
    last observed discharge cannot be confirmed as non-readmissions — Synthea simply
    stopped generating data.  Labelling them False would inject false negatives.
    These encounters receive readmitted_30d=NaN and are excluded from the ML feature
    table by the downstream dropna() in build_encounter_ml_features.
    """
    df = df_enc.copy().sort_values(["patient_id", "admission_date"])

    inpatient_mask = df["encounter_class"].isin(INPATIENT_CLASSES)
    df_ip = df.loc[inpatient_mask, ["patient_id", "encounter_id", "admission_date", "discharge_date"]].copy()
    df_ip = df_ip.sort_values(["patient_id", "admission_date"])

    # Next inpatient admission date for each encounter within the same patient
    df_ip["next_admit"] = df_ip.groupby("patient_id")["admission_date"].shift(-1)
    df_ip["days_to_next"] = (
        (df_ip["next_admit"] - df_ip["discharge_date"]).dt.total_seconds() / 86_400
    )

    readmitted_ids = set(
        df_ip.loc[
            df_ip["days_to_next"].between(0, 30, inclusive="both"),
            "encounter_id",
        ]
    )

    # Censoring boundary: any discharge within 30 days of the dataset's last discharge.
    # If we observed the readmission it's confirmed True regardless of proximity to cutoff.
    max_discharge = df_ip["discharge_date"].max()
    censored_ids = set(
        df_ip.loc[
            (df_ip["discharge_date"] > max_discharge - pd.Timedelta(days=30)) &
            ~df_ip["encounter_id"].isin(readmitted_ids),
            "encounter_id",
        ]
    )

    # Assign labels: True = confirmed readmission, False = confirmed negative,
    # NaN = censored (excluded from ML training).
    label_map = {eid: True for eid in readmitted_ids}
    label_map.update({
        eid: False
        for eid in df_ip["encounter_id"]
        if eid not in readmitted_ids and eid not in censored_ids
    })
    # Censored encounter_ids are absent from label_map → map produces NaN

    df["readmitted_30d"] = df["encounter_id"].map(label_map)
    # Non-inpatient encounters are irrelevant to ML — give them False so the
    # fact_encounters table has no nulls outside the inpatient population.
    df.loc[~inpatient_mask, "readmitted_30d"] = False

    n_ip = inpatient_mask.sum()
    logger.info(
        "Readmission labels: %d positive | %d censored (will be excluded) | "
        "%d confirmed negative  (out of %d inpatient encounters)",
        len(readmitted_ids), len(censored_ids),
        n_ip - len(readmitted_ids) - len(censored_ids), n_ip,
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

    feat = df_patients[["patient_id", "birth_date", "gender", "insurance_type"]].copy()
    feat["age"] = feat["birth_date"].apply(lambda d: _age_on(d, today))
    feat["age_bucket"] = feat["age"].apply(_age_bucket)
    feat["gender_encoded"] = (feat["gender"] == "male").astype(int)
    feat["insurance_risk_tier"] = feat["insurance_type"].apply(
        lambda x: INSURANCE_RISK_TIERS.get(x, DEFAULT_INSURANCE_TIER)
    )

    # Sort before aggregation so "last" gives the most recent encounter's LOS
    enc_sorted = df_encounters.sort_values(["patient_id", "admission_date"])
    enc_agg = (
        enc_sorted.groupby("patient_id")
        .agg(
            total_encounters=("encounter_id", "count"),
            last_los_days=("length_of_stay_days", "last"),
        )
        .reset_index()
    )
    feat = feat.merge(enc_agg, on="patient_id", how="left")

    # prior_admissions_12m: count inpatient admissions in the 12 months before
    # each patient's LAST admission, not relative to wall-clock "now".
    # Using wall-clock time would yield all-zero since Synthea data ends before 2026.
    df_ip = df_encounters[df_encounters["encounter_class"].isin(INPATIENT_CLASSES)].copy()
    last_admit = (
        df_ip.groupby("patient_id")["admission_date"].max()
        .rename("last_admit").reset_index()
    )
    df_ip = df_ip.merge(last_admit, on="patient_id", how="left")
    prior_12m = (
        df_ip[
            (df_ip["admission_date"] < df_ip["last_admit"]) &
            (df_ip["last_admit"] - df_ip["admission_date"] <= pd.Timedelta(days=365))
        ]
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

    # Synthea CSVs carry no reference ranges, so compute population z-scores per LOINC code.
    # Same approach as build_encounter_ml_features — consistent across both feature tables.
    # "laboratory" only: exclude vital-signs (heart rate, BP, temp) which are not lab results.
    lab_num = df_labs[df_labs["category"] == "laboratory"].dropna(subset=["value_numeric", "loinc_code"]).copy()
    pop_stats = (
        lab_num.groupby("loinc_code")["value_numeric"]
        .agg(pop_mean="mean", pop_std="std")
        .reset_index()
    )
    pop_stats["pop_std"] = pop_stats["pop_std"].fillna(1.0).clip(lower=1e-6)
    lab_num = lab_num.merge(pop_stats, on="loinc_code", how="left")
    lab_num["z_score"] = (lab_num["value_numeric"] - lab_num["pop_mean"]) / lab_num["pop_std"]
    lab_dev = (
        lab_num.groupby("patient_id")["z_score"]
        .apply(lambda s: s.abs().mean())
        .rename("avg_lab_deviation")
        .reset_index()
    )
    feat = feat.merge(lab_dev, on="patient_id", how="left")

    med_count = (
        df_medications.groupby("patient_id")["medication_id"]
        .count()
        .rename("medication_count")
        .reset_index()
    )
    feat = feat.merge(med_count, on="patient_id", how="left")

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


# ─── Encounter-level ML features (no leakage) ────────────────────────────────
# INPATIENT_CLASSES is imported from etl.ml_config — single source of truth shared
# with add_readmission_label (above) and pipeline.py (DB load filter).

CHRONIC_ICD_MAP = {
    "has_heart_failure": ["I50"],
    "has_diabetes":      ["E10", "E11"],
    "has_copd":          ["J44"],
    "has_ckd":           ["N18"],
    "has_hypertension":  ["I10"],
}

# Synthea's conditions.csv CODE column contains SNOMED CT codes (e.g. 44054006),
# not ICD-10.  str[:3] on a SNOMED code gives "440", which never matches "E10".
# Result: all ICD prefix checks return zero matches → all flags are False.
#
# Fix: also check the DESCRIPTION column using plain-English keywords.
# The union of ICD-prefix matches and description matches is used for the flag,
# so this works correctly for both Synthea (SNOMED) and real ICD-10 coded data.
CHRONIC_DESCRIPTION_KEYWORDS = {
    "has_heart_failure": ["heart failure"],
    "has_diabetes":      ["diabetes mellitus", "type 1 diabetes", "type 2 diabetes"],
    "has_copd":          ["chronic obstructive", "copd"],
    "has_ckd":           ["chronic kidney disease"],
    "has_hypertension":  ["hypertension"],
}

ENCOUNTER_CLASS_ENCODING = {
    "wellness": 0, "ambulatory": 0, "outpatient": 0,
    "urgentcare": 1, "emergency": 1, "EMER": 1,
    "inpatient": 2, "IMP": 2,
}


def build_encounter_ml_features(
    df_patients: pd.DataFrame,
    df_encounters: pd.DataFrame,
    df_diagnoses: pd.DataFrame,
    df_labs: pd.DataFrame,
    df_medications: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per inpatient/emergency encounter.
    All history features computed from data BEFORE admission date (no leakage).
    Vectorized — no Python-level row loops.
    """
    df_ip = df_encounters[
        df_encounters["encounter_class"].isin(INPATIENT_CLASSES)
    ].copy()
    df_ip = df_ip.dropna(subset=["admission_date", "readmitted_30d"]).reset_index(drop=True)
    logger.info("Encounter-level dataset: %d inpatient/emergency encounters", len(df_ip))

    # ── Patient demographics ──────────────────────────────────────────────────
    pat = df_patients[["patient_id", "birth_date", "gender", "insurance_type"]].copy()
    pat["birth_date"] = pd.to_datetime(pat["birth_date"], errors="coerce")
    df_ip = df_ip.merge(pat, on="patient_id", how="left")

    # tz_convert(None) strips UTC offset without changing the underlying time
    df_ip["age_at_admission"] = (
        (df_ip["admission_date"].dt.tz_convert(None) - df_ip["birth_date"])
        .dt.days // 365
    ).clip(lower=0)

    df_ip["gender_encoded"] = (df_ip["gender"].str.lower() == "male").astype(int)
    df_ip["insurance_risk_tier"] = df_ip["insurance_type"].apply(
        lambda x: INSURANCE_RISK_TIERS.get(x, DEFAULT_INSURANCE_TIER)
    )
    df_ip["encounter_class_encoded"] = (
        df_ip["encounter_class"].map(ENCOUNTER_CLASS_ENCODING).fillna(1).astype(int)
    )

    # ── Diagnoses this encounter ──────────────────────────────────────────────
    diag_enc = (
        df_diagnoses.groupby("encounter_id")
        .agg(num_diagnoses_this_visit=("icd_code", "count"))
        .reset_index()
    )
    df_ip = df_ip.merge(diag_enc, on="encounter_id", how="left")
    df_ip["num_diagnoses_this_visit"] = df_ip["num_diagnoses_this_visit"].fillna(0).astype(int)

    # Chronic flags are computed later (after prior_diag is built) so they can
    # include patient history — see "Chronic condition flags" section below.

    # ── Labs this encounter ───────────────────────────────────────────────────
    # Population z-scores: z = (value - LOINC_mean) / LOINC_std.
    # Abnormal = |z| > 2.  Filter to "laboratory" only (not vital-signs).
    #
    # Leakage-free normalisation: LOINC mean/std are fitted on TRAINING encounters
    # only, then applied to all encounters.  The patient split reproduces
    # _patient_split() in models/train.py exactly — same sklearn function, same
    # parameters (test_size=0.2, random_state=42), same pandas-sorted patient order
    # — so the training encounter set here matches the one used by the model.
    # Storing these values in the DB means ml_encounter_features is correct for
    # Phase 3 RAG queries and no post-ETL recomputation step is needed in training.
    _pat_labels = (
        df_ip.groupby("patient_id")["readmitted_30d"]
        .any()
        .astype(int)
        .reset_index()
        .rename(columns={"readmitted_30d": "has_any_readmit"})
    )
    _train_pats, _ = _skl_tts(
        _pat_labels["patient_id"],
        test_size=SPLIT_TEST_SIZE,
        stratify=_pat_labels["has_any_readmit"],
        random_state=SPLIT_RANDOM_STATE,
    )
    _train_enc_ids = set(
        df_ip.loc[df_ip["patient_id"].isin(set(_train_pats)), "encounter_id"]
    )
    logger.info(
        "Lab normaliser: fitted on %d training encounters (%d test)",
        len(_train_enc_ids), len(df_ip) - len(_train_enc_ids),
    )

    lab_num = df_labs[df_labs["category"] == "laboratory"].dropna(subset=["value_numeric", "loinc_code"]).copy()

    # Raw lab count computed BEFORE LOINC filtering so it reflects the true number
    # of lab observations, not just those whose LOINC code appeared in training.
    # If counted after filtering, test encounters with rare LOINC codes get an
    # artificially lower count than training encounters — the feature means
    # different things in train vs test.
    raw_lab_count = (
        lab_num.groupby("encounter_id")["value_numeric"]
        .count()
        .rename("num_labs_this_visit")
        .reset_index()
    )

    # LOINC population stats fitted on training encounters only (no leakage).
    # LOINC codes not seen in training are dropped — their population distribution
    # is unknown and cannot produce a valid z-score.
    pop_stats = (
        lab_num[lab_num["encounter_id"].isin(_train_enc_ids)]
        .groupby("loinc_code")["value_numeric"]
        .agg(pop_mean="mean", pop_std="std")
        .reset_index()
    )
    pop_stats["pop_std"] = pop_stats["pop_std"].fillna(1.0).clip(lower=1e-6)
    lab_num = lab_num.merge(pop_stats, on="loinc_code", how="left")
    lab_num = lab_num.dropna(subset=["pop_mean"])  # drop LOINC codes unseen in training
    lab_num["z_score"] = (lab_num["value_numeric"] - lab_num["pop_mean"]) / lab_num["pop_std"]
    lab_num["is_abnormal"] = lab_num["z_score"].abs() > 2.0

    # Anomaly features from LOINC-normalised labs only (excludes rare LOINC codes).
    # num_labs_this_visit is joined separately from the unfiltered raw_lab_count above.
    lab_enc_agg = (
        lab_num.groupby("encounter_id")
        .agg(
            num_abnormal_labs_this_visit=("is_abnormal", "sum"),
            avg_lab_deviation_this_visit=("z_score", lambda s: s.abs().mean()),
        )
        .reset_index()
    )
    df_ip = df_ip.merge(raw_lab_count, on="encounter_id", how="left")
    df_ip = df_ip.merge(lab_enc_agg, on="encounter_id", how="left")
    df_ip["num_labs_this_visit"] = df_ip["num_labs_this_visit"].fillna(0).astype(int)
    df_ip["num_abnormal_labs_this_visit"] = df_ip["num_abnormal_labs_this_visit"].fillna(0).astype(int)
    df_ip["avg_lab_deviation_this_visit"] = df_ip["avg_lab_deviation_this_visit"].fillna(0.0)

    # ── Medications this encounter ────────────────────────────────────────────
    med_enc = (
        df_medications.groupby("encounter_id")
        .agg(num_meds_this_visit=("medication_id", "count"))
        .reset_index()
    )
    df_ip = df_ip.merge(med_enc, on="encounter_id", how="left")
    df_ip["num_meds_this_visit"] = df_ip["num_meds_this_visit"].fillna(0).astype(int)

    # ── Patient history BEFORE this encounter (vectorized, no leakage) ────────
    # Self-merge all inpatient encounters on patient_id, then filter prior < target.
    # Avoids O(n²) Python loops — all filtering done in pandas C layer.
    all_ip = df_encounters[
        df_encounters["encounter_class"].isin(INPATIENT_CLASSES)
    ][["patient_id", "encounter_id", "admission_date"]].copy()

    target = df_ip[["encounter_id", "patient_id", "admission_date"]].copy()

    prior = target.merge(
        all_ip.rename(columns={"encounter_id": "prior_enc_id", "admission_date": "prior_admit"}),
        on="patient_id",
        how="left",
    )
    prior = prior[prior["prior_admit"] < prior["admission_date"]].copy()

    prior_total = (
        prior.groupby("encounter_id").size()
        .rename("prior_admissions_total").reset_index()
    )
    prior_12m = (
        prior[prior["admission_date"] - prior["prior_admit"] <= pd.Timedelta(days=365)]
        .groupby("encounter_id").size()
        .rename("prior_admissions_12m").reset_index()
    )
    prior_6m = (
        prior[prior["admission_date"] - prior["prior_admit"] <= pd.Timedelta(days=180)]
        .groupby("encounter_id").size()
        .rename("prior_admissions_6m").reset_index()
    )

    # Days since most recent prior admission.
    # NaN for first admissions — is_first_admission (below) captures that case cleanly.
    # We do NOT use a 999 sentinel: StandardScaler would treat it as a real high value
    # and distort the feature distribution.  Median imputation (in the train pipeline)
    # handles the NaNs without leakage.
    prev_admit = prior.groupby("encounter_id")["prior_admit"].max().reset_index()
    prev_admit.columns = ["encounter_id", "prev_admit_date"]
    prev_admit = target[["encounter_id", "admission_date"]].merge(prev_admit, on="encounter_id", how="left")
    prev_admit["days_since_previous_visit"] = (
        (prev_admit["admission_date"] - prev_admit["prev_admit_date"]).dt.days
        # float64 with NaN for first admissions
    )

    for stat_df in [prior_total, prior_12m, prior_6m]:
        df_ip = df_ip.merge(stat_df, on="encounter_id", how="left")
    df_ip = df_ip.merge(
        prev_admit[["encounter_id", "days_since_previous_visit"]], on="encounter_id", how="left"
    )

    df_ip["prior_admissions_total"] = df_ip["prior_admissions_total"].fillna(0).astype(int)
    df_ip["prior_admissions_12m"]   = df_ip["prior_admissions_12m"].fillna(0).astype(int)
    df_ip["prior_admissions_6m"]    = df_ip["prior_admissions_6m"].fillna(0).astype(int)
    # Nullable Int64: NaN → NULL in DB → median-imputed in training pipeline.
    df_ip["days_since_previous_visit"] = df_ip["days_since_previous_visit"].astype("Int64")
    df_ip["is_first_admission"]        = (df_ip["prior_admissions_total"] == 0).astype(int)

    # ── Comorbidity count from diagnoses BEFORE this encounter (vectorized) ────
    diag_with_admit = df_diagnoses.merge(
        df_encounters[["encounter_id", "admission_date"]].rename(
            columns={"encounter_id": "diag_enc_id", "admission_date": "diag_enc_admit"}
        ),
        left_on="encounter_id", right_on="diag_enc_id",
        how="left",
    )
    comorbidity_cross = target.merge(
        diag_with_admit[["patient_id", "diag_enc_admit", "icd_code", "icd_description"]],
        on="patient_id",
        how="left",
    )
    prior_diag = comorbidity_cross[
        comorbidity_cross["diag_enc_admit"] < comorbidity_cross["admission_date"]
    ]
    comorbidity_count = (
        prior_diag.groupby("encounter_id")["icd_code"].nunique()
        .rename("comorbidity_count_prior").reset_index()
    )
    df_ip = df_ip.merge(comorbidity_count, on="encounter_id", how="left")
    df_ip["comorbidity_count_prior"] = df_ip["comorbidity_count_prior"].fillna(0).astype(int)

    # ── Chronic condition flags: current encounter OR prior patient history ────
    # Chronic conditions in Synthea are coded at the encounter where they were
    # first diagnosed, NOT re-coded at every subsequent encounter.  A patient with
    # heart failure admitted for pneumonia has no I50 code on the pneumonia encounter
    # — so a per-encounter check produces high false-negative rates.
    # Correct definition: flag = True if coded at THIS encounter OR at any prior
    # encounter whose admission date < this admission date (no leakage).
    #
    # Detection uses TWO complementary methods unioned together:
    #   1. ICD-10 prefix match on icd_code  — works for real clinical data
    #   2. Keyword match on icd_description — works for Synthea (SNOMED CT codes)
    for flag, prefixes in CHRONIC_ICD_MAP.items():
        prefix_set = set(prefixes)
        keywords   = CHRONIC_DESCRIPTION_KEYWORDS.get(flag, [])
        kw_pattern = "|".join(keywords) if keywords else None

        def _flagged_by_code(df: pd.DataFrame) -> set:
            return set(
                df.loc[df["icd_code"].str[:3].isin(prefix_set), "encounter_id"].dropna()
            )

        def _flagged_by_desc(df: pd.DataFrame) -> set:
            if kw_pattern is None or "icd_description" not in df.columns:
                return set()
            return set(
                df.loc[
                    df["icd_description"].str.lower().str.contains(kw_pattern, na=False),
                    "encounter_id",
                ].dropna()
            )

        current_flagged = _flagged_by_code(df_diagnoses) | _flagged_by_desc(df_diagnoses)
        prior_flagged   = _flagged_by_code(prior_diag)   | _flagged_by_desc(prior_diag)
        df_ip[flag] = df_ip["encounter_id"].isin(current_flagged | prior_flagged)

    # ── Select final columns ──────────────────────────────────────────────────
    keep = [
        "encounter_id", "patient_id", "readmitted_30d",
        "age_at_admission", "gender_encoded", "insurance_risk_tier",
        "encounter_class_encoded", "length_of_stay_days", "total_claim_cost",
        "num_diagnoses_this_visit", "has_heart_failure", "has_diabetes",
        "has_copd", "has_ckd", "has_hypertension",
        "num_labs_this_visit", "num_abnormal_labs_this_visit", "avg_lab_deviation_this_visit",
        "num_meds_this_visit",
        "prior_admissions_6m", "prior_admissions_12m", "prior_admissions_total",
        "days_since_previous_visit", "is_first_admission", "comorbidity_count_prior",
    ]
    df_ip = df_ip[[c for c in keep if c in df_ip.columns]]
    df_ip["total_claim_cost"] = df_ip["total_claim_cost"].fillna(0.0)
    # length_of_stay_days intentionally left as NaN (→ NULL in DB) when discharge_date
    # is missing.  Filling with 0.0 would conflate "missing discharge" with a genuine
    # same-day discharge and defeat the LOS regressor's "> 0" filter.
    # The readmission classifier's SimpleImputer(strategy="median") handles NULL at
    # training time without leakage.

    logger.info(
        "Encounter features: %d rows, %d columns, %.1f%% readmitted",
        len(df_ip), len(df_ip.columns),
        df_ip["readmitted_30d"].mean() * 100,
    )
    return df_ip
