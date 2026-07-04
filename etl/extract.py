import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _csv(csv_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    path = csv_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False, **kwargs)
    logger.info("Loaded %s: %d rows", name, len(df))
    return df


def extract_patients(csv_dir: Path) -> pd.DataFrame:
    df = _csv(csv_dir, "patients.csv")
    payers = _csv(csv_dir, "payers.csv", usecols=["Id", "NAME"])
    transitions = _csv(
        csv_dir, "payer_transitions.csv",
        usecols=["PATIENT", "PAYER", "START_DATE", "END_DATE"],
    )

    transitions["START_DATE"] = pd.to_datetime(transitions["START_DATE"], errors="coerce", utc=True)
    latest = (
        transitions.sort_values("START_DATE")
        .groupby("PATIENT")["PAYER"]
        .last()
        .reset_index()
        .merge(payers.rename(columns={"Id": "PAYER", "NAME": "insurance_type"}), on="PAYER", how="left")
    )

    df = df.merge(latest[["PATIENT", "insurance_type"]], left_on="Id", right_on="PATIENT", how="left")
    df["insurance_type"] = df["insurance_type"].fillna("NO_INSURANCE")

    out = pd.DataFrame({
        "patient_id":     df["Id"],
        "first_name":     df["FIRST"],
        "last_name":      df["LAST"],
        "birth_date":     df["BIRTHDATE"],
        "gender":         df["GENDER"].str.lower(),
        "race":           df["RACE"],
        "ethnicity":      df["ETHNICITY"],
        "address_city":   df["CITY"],
        "address_state":  df["STATE"],
        "zip_code":       df["ZIP"].astype(str).str.zfill(5),
        "insurance_type": df["insurance_type"],
    })
    return out.drop_duplicates(subset="patient_id").reset_index(drop=True)


def extract_encounters(csv_dir: Path) -> pd.DataFrame:
    df = _csv(csv_dir, "encounters.csv")
    out = pd.DataFrame({
        "encounter_id":          df["Id"],
        "patient_id":            df["PATIENT"],
        "encounter_class":       df["ENCOUNTERCLASS"],
        "encounter_type":        df["DESCRIPTION"],
        "admission_date":        df["START"],
        "discharge_date":        df["STOP"],
        "primary_diagnosis_code": df.get("REASONCODE", pd.Series(dtype=str)),
        "primary_diagnosis_desc": df.get("REASONDESCRIPTION", pd.Series(dtype=str)),
        "provider_org":          df.get("ORGANIZATION", pd.Series(dtype=str)),
        "total_claim_cost":      pd.to_numeric(df.get("TOTAL_CLAIM_COST"), errors="coerce"),
        "payer_coverage":        pd.to_numeric(df.get("PAYER_COVERAGE"), errors="coerce"),
    })
    return out.drop_duplicates(subset="encounter_id").reset_index(drop=True)


def extract_conditions(csv_dir: Path) -> pd.DataFrame:
    df = _csv(csv_dir, "conditions.csv")
    df["diagnosis_id"] = (
        df["PATIENT"].astype(str) + "_" +
        df["ENCOUNTER"].fillna("").astype(str) + "_" +
        df["CODE"].astype(str) + "_" +
        df["START"].astype(str)
    )
    out = pd.DataFrame({
        "diagnosis_id":    df["diagnosis_id"],
        "encounter_id":    df["ENCOUNTER"],
        "patient_id":      df["PATIENT"],
        "icd_code":        df["CODE"].astype(str),
        "icd_description": df["DESCRIPTION"],
        "onset_date":      df["START"],
        "abatement_date":  df.get("STOP", pd.Series(dtype=str)),
        "clinical_status": df["STOP"].apply(
            lambda s: "resolved" if pd.notna(s) and s != "" else "active"
        ),
    })
    return out.drop_duplicates(subset="diagnosis_id").reset_index(drop=True)


def extract_observations(csv_dir: Path) -> pd.DataFrame:
    df = _csv(csv_dir, "observations.csv")
    df = df[df["CATEGORY"].isin(["laboratory", "vital-signs"])].copy()
    df["lab_id"] = (
        df["PATIENT"].astype(str) + "_" +
        df["ENCOUNTER"].fillna("").astype(str) + "_" +
        df["CODE"].astype(str) + "_" +
        df["DATE"].astype(str)
    )

    numeric_mask = df["TYPE"] == "numeric"
    out = pd.DataFrame({
        "lab_id":           df["lab_id"],
        "encounter_id":     df["ENCOUNTER"],
        "patient_id":       df["PATIENT"],
        "category":         df["CATEGORY"],
        "loinc_code":       df["CODE"].astype(str),
        "lab_name":         df["DESCRIPTION"],
        "value_numeric":    pd.to_numeric(df["VALUE"].where(numeric_mask), errors="coerce"),
        "value_string":     df["VALUE"].where(~numeric_mask),
        "unit":             df.get("UNITS", pd.Series(dtype=str)),
        "reference_low":    None,
        "reference_high":   None,
        "observation_date": df["DATE"],
    })
    return out.drop_duplicates(subset="lab_id").reset_index(drop=True)


def extract_medications(csv_dir: Path) -> pd.DataFrame:
    df = _csv(csv_dir, "medications.csv")
    df["medication_id"] = (
        df["PATIENT"].astype(str) + "_" +
        df["ENCOUNTER"].fillna("").astype(str) + "_" +
        df["CODE"].astype(str) + "_" +
        df["START"].astype(str)
    )
    out = pd.DataFrame({
        "medication_id":   df["medication_id"],
        "encounter_id":    df["ENCOUNTER"],
        "patient_id":      df["PATIENT"],
        "medication_code": df["CODE"].astype(str),
        "medication_name": df["DESCRIPTION"],
        "start_date":      df["START"].astype(str).str[:10],
        "end_date":        df["STOP"].astype(str).str[:10].replace("nan", None),
        "status":          df["STOP"].apply(lambda s: "stopped" if pd.notna(s) and s != "" else "active"),
    })
    return out.drop_duplicates(subset="medication_id").reset_index(drop=True)


def extract_all(csv_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Extract all tables from Synthea CSV output directory."""
    csv_dir = Path(csv_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    return {
        "patients":    extract_patients(csv_dir),
        "encounters":  extract_encounters(csv_dir),
        "conditions":  extract_conditions(csv_dir),
        "observations": extract_observations(csv_dir),
        "medications": extract_medications(csv_dir),
    }
