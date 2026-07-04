import logging
import os
import textwrap
from pathlib import Path

import pandas as pd
import sqlalchemy
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv()

OUTPUT_CSV = Path("data/synthea_notes.csv")


def _engine():
    url = os.environ["DATABASE_URL"].replace("postgres://", "postgresql://")
    if "sslmode" not in url:
        url += "?sslmode=require"
    return sqlalchemy.create_engine(url)


def _load_tables(engine) -> dict[str, pd.DataFrame]:
    logger.info("Loading tables from Supabase...")
    tables = {}
    tables["patients"] = pd.read_sql(
        "SELECT patient_id, first_name, last_name, birth_date, gender, race FROM dim_patient",
        engine,
    )
    tables["encounters"] = pd.read_sql(
        """SELECT encounter_id, patient_id, admission_date, discharge_date,
                  length_of_stay_days, readmitted_30d,
                  primary_diagnosis_desc, encounter_class, total_claim_cost
           FROM fact_encounters
           WHERE encounter_class IN ('inpatient','IMP','emergency','EMER')
             AND discharge_date IS NOT NULL""",
        engine,
    )
    tables["diagnoses"] = pd.read_sql(
        "SELECT encounter_id, icd_description, clinical_status FROM dim_diagnosis",
        engine,
    )
    tables["medications"] = pd.read_sql(
        "SELECT encounter_id, medication_name, start_date FROM dim_medication",
        engine,
    )
    tables["labs"] = pd.read_sql(
        """SELECT encounter_id, lab_name, value_numeric, unit, reference_low, reference_high
           FROM dim_lab_result
           WHERE value_numeric IS NOT NULL""",
        engine,
    )
    for name, df in tables.items():
        logger.info("  %-12s: %d rows", name, len(df))
    return tables


def _admission_date(enc: pd.Series) -> str:
    try:
        return pd.Timestamp(enc["admission_date"]).strftime("%Y-%m-%d")
    except Exception:
        return "Unknown"


def _discharge_date(enc: pd.Series) -> str:
    try:
        return pd.Timestamp(enc["discharge_date"]).strftime("%Y-%m-%d")
    except Exception:
        return "Unknown"


def _age_at_admission(enc: pd.Series, pat: pd.Series) -> str:
    try:
        dob = pd.Timestamp(pat["birth_date"])
        adm = pd.Timestamp(enc["admission_date"])
        return str(int((adm - dob).days / 365.25))
    except Exception:
        return "Unknown"


def _lab_summary(enc_labs: pd.DataFrame) -> str:
    if enc_labs.empty:
        return "No laboratory results documented during this admission."
    lines = []
    for _, lab in enc_labs.head(12).iterrows():
        flag = ""
        if pd.notna(lab.get("reference_low")) and pd.notna(lab.get("value_numeric")):
            if lab["value_numeric"] < lab["reference_low"]:
                flag = " (LOW)"
            elif pd.notna(lab.get("reference_high")) and lab["value_numeric"] > lab["reference_high"]:
                flag = " (HIGH)"
        unit = f" {lab['unit']}" if pd.notna(lab.get("unit")) else ""
        lines.append(f"  {lab['lab_name']}: {lab['value_numeric']:.1f}{unit}{flag}")
    return "\n".join(lines)


def _med_list(enc_meds: pd.DataFrame) -> str:
    if enc_meds.empty:
        return "No medications recorded during this admission."
    names = enc_meds["medication_name"].dropna().unique()
    return "\n".join(f"  - {n}" for n in names[:20])


def _diag_list(enc_diags: pd.DataFrame) -> str:
    if enc_diags.empty:
        return "No diagnoses documented."
    active = enc_diags[enc_diags["clinical_status"].isin(["active", "confirmed", None])]["icd_description"]
    names = active.dropna().unique()
    if len(names) == 0:
        names = enc_diags["icd_description"].dropna().unique()
    return "\n".join(f"  - {d}" for d in names[:15])


def build_discharge_summary(
    enc: pd.Series,
    pat: pd.Series,
    enc_diags: pd.DataFrame,
    enc_meds: pd.DataFrame,
    enc_labs: pd.DataFrame,
) -> str:
    los = enc.get("length_of_stay_days", "Unknown")
    los_str = f"{los:.1f}" if pd.notna(los) else "Unknown"
    readmit = enc.get("readmitted_30d")
    readmit_str = "Yes" if readmit else "No" if readmit is False else "Unknown"

    return textwrap.dedent(f"""
    DISCHARGE SUMMARY
    =================
    Patient: {pat.get('first_name', '[REDACTED]')} {pat.get('last_name', '[REDACTED]')}
    Patient ID: {enc['patient_id']}
    Gender: {pat.get('gender', 'Unknown')}   Race: {pat.get('race', 'Unknown')}
    Age at Admission: {_age_at_admission(enc, pat)} years

    ADMISSION DATE: {_admission_date(enc)}
    DISCHARGE DATE: {_discharge_date(enc)}
    LENGTH OF STAY: {los_str} days
    ENCOUNTER CLASS: {enc.get('encounter_class', 'Unknown')}
    READMITTED WITHIN 30 DAYS: {readmit_str}

    PRIMARY DIAGNOSIS:
      {enc.get('primary_diagnosis_desc', 'Not documented')}

    DIAGNOSES THIS ADMISSION:
    {_diag_list(enc_diags)}

    LABORATORY RESULTS:
    {_lab_summary(enc_labs)}

    MEDICATIONS DURING ADMISSION:
    {_med_list(enc_meds)}

    DISPOSITION:
    Patient was discharged in stable condition. Follow-up arranged as clinically indicated.
    Total claim cost: ${enc.get('total_claim_cost', 0) or 0:.2f}
    """).strip()


def main() -> None:
    engine = _engine()
    tables = _load_tables(engine)
    engine.dispose()

    patients   = tables["patients"].set_index("patient_id")
    encounters = tables["encounters"]

    diags_by_enc = {k: v for k, v in tables["diagnoses"].groupby("encounter_id")}
    meds_by_enc  = {k: v for k, v in tables["medications"].groupby("encounter_id")}
    labs_by_enc  = {k: v for k, v in tables["labs"].groupby("encounter_id")}
    logger.info("Pre-grouped lookups ready; generating notes...")

    _empty = pd.DataFrame()
    rows = []
    row_id = 1
    for _, enc in encounters.iterrows():
        enc_id = enc["encounter_id"]
        pat_id = enc["patient_id"]

        if pat_id not in patients.index:
            continue

        pat = patients.loc[pat_id]
        enc_diags = diags_by_enc.get(enc_id, _empty)
        enc_meds  = meds_by_enc.get(enc_id, _empty)
        enc_labs  = labs_by_enc.get(enc_id, _empty)

        note_text = build_discharge_summary(enc, pat, enc_diags, enc_meds, enc_labs)

        rows.append({
            "row_id":      row_id,
            "subject_id":  pat_id,
            "hadm_id":     enc_id,
            "chartdate":   _discharge_date(enc),
            "charttime":   "",
            "storetime":   "",
            "category":    "Discharge summary",
            "description": "Report",
            "cgid":        "",
            "iserror":     "",
            "text":        note_text,
        })
        row_id += 1

    df_notes = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_notes.to_csv(str(OUTPUT_CSV), index=False)
    logger.info(
        "Generated %d discharge notes for %d patients → %s",
        len(df_notes),
        df_notes["subject_id"].nunique(),
        OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
