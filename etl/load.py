"""
Bulk-loads transformed DataFrames into PostgreSQL using psycopg2.
Uses COPY for speed on large tables and upserts for idempotency.
"""

import io
import logging
import os

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)


def get_connection(dsn: str | None = None):
    dsn = dsn or os.environ["DATABASE_URL"]
    # Supabase (and most cloud Postgres) requires SSL
    if "sslmode" not in dsn:
        dsn += "?sslmode=require"
    return psycopg2.connect(dsn)


def _df_to_copy_buffer(df: pd.DataFrame) -> io.StringIO:
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    return buf


def _bulk_copy(conn, df: pd.DataFrame, table: str, columns: list[str]) -> int:
    """COPY FROM for maximum throughput. Returns rows copied."""
    subset = df[columns].copy()
    buf = _df_to_copy_buffer(subset)
    with conn.cursor() as cur:
        cur.copy_expert(
            f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH CSV NULL '\\N'",
            buf,
        )
    return len(subset)


def _safe_val(v):
    """Convert NaT/NaN to NULL and numpy types to native Python for psycopg2."""
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _upsert(conn, df: pd.DataFrame, table: str, columns: list[str], pk: str) -> int:
    """INSERT ... ON CONFLICT DO NOTHING for idempotent loads."""
    records = [
        tuple(_safe_val(v) for v in row)
        for row in df[columns].itertuples(index=False)
    ]
    col_str = ", ".join(columns)
    placeholders = "(" + ", ".join(["%s"] * len(columns)) + ")"
    sql = (
        f"INSERT INTO {table} ({col_str}) VALUES %s "
        f"ON CONFLICT ({pk}) DO NOTHING"
    )
    with conn.cursor() as cur:
        execute_values(cur, sql, records, template=placeholders, page_size=1000)
    return len(records)


def apply_schema(conn) -> None:
    """Create tables if they don't exist yet."""
    schema_path = os.path.join(os.path.dirname(__file__), "..", "sql", "schema.sql")
    with open(schema_path, encoding="utf-8") as f:
        ddl = f.read()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    logger.info("Schema applied")


def load_patients(conn, df: pd.DataFrame) -> int:
    cols = [
        "patient_id", "first_name", "last_name", "birth_date",
        "gender", "race", "ethnicity", "address_city",
        "address_state", "zip_code", "insurance_type",
    ]
    n = _upsert(conn, df, "dim_patient", cols, "patient_id")
    conn.commit()
    logger.info("Loaded %d patients", n)
    return n


def load_encounters(conn, df: pd.DataFrame) -> int:
    # Convert timezone-aware timestamps to naive UTC strings for psycopg2
    df = df.copy()
    for col in ("admission_date", "discharge_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

    cols = [
        "encounter_id", "patient_id", "encounter_class", "encounter_type",
        "admission_date", "discharge_date", "length_of_stay_days",
        "readmitted_30d", "primary_diagnosis_code", "primary_diagnosis_desc",
        "provider_org", "total_claim_cost", "payer_coverage",
    ]
    # Only include columns that actually exist in df
    cols = [c for c in cols if c in df.columns]
    n = _upsert(conn, df, "fact_encounters", cols, "encounter_id")
    conn.commit()
    logger.info("Loaded %d encounters", n)
    return n


def load_diagnoses(conn, df: pd.DataFrame) -> int:
    cols = [
        "diagnosis_id", "encounter_id", "patient_id",
        "icd_code", "icd_description", "onset_date",
        "abatement_date", "clinical_status",
    ]
    cols = [c for c in cols if c in df.columns]
    n = _upsert(conn, df, "dim_diagnosis", cols, "diagnosis_id")
    conn.commit()
    logger.info("Loaded %d diagnoses", n)
    return n


def load_labs(conn, df: pd.DataFrame) -> int:
    df = df.copy()
    if "observation_date" in df.columns:
        df["observation_date"] = pd.to_datetime(df["observation_date"], utc=True).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    cols = [
        "lab_id", "encounter_id", "patient_id", "loinc_code", "lab_name",
        "value_numeric", "value_string", "unit",
        "reference_low", "reference_high", "observation_date",
    ]
    cols = [c for c in cols if c in df.columns]
    n = _upsert(conn, df, "dim_lab_result", cols, "lab_id")
    conn.commit()
    logger.info("Loaded %d lab results", n)
    return n


def load_medications(conn, df: pd.DataFrame) -> int:
    cols = [
        "medication_id", "encounter_id", "patient_id",
        "medication_code", "medication_name",
        "start_date", "end_date", "status",
    ]
    cols = [c for c in cols if c in df.columns]
    n = _upsert(conn, df, "dim_medication", cols, "medication_id")
    conn.commit()
    logger.info("Loaded %d medications", n)
    return n


def load_ml_features(conn, df: pd.DataFrame) -> int:
    cols = [
        "patient_id", "age", "age_bucket", "gender_encoded",
        "comorbidity_count", "chronic_condition_count", "medication_count",
        "prior_admissions_12m", "prior_admissions_total",
        "days_since_last_visit", "avg_lab_deviation",
        "insurance_risk_tier", "total_encounters", "last_los_days",
    ]
    cols = [c for c in cols if c in df.columns]
    n = _upsert(conn, df, "ml_features", cols, "patient_id")
    conn.commit()
    logger.info("Loaded %d ML feature rows", n)
    return n
