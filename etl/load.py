"""
Bulk-loads transformed DataFrames into PostgreSQL using psycopg2.

Small tables  (<50k rows) : _upsert      — execute_values, single commit
Large tables (>=50k rows) : _bulk_upsert — COPY → temp table → ON CONFLICT,
                             committed every CHUNK_SIZE rows to prevent
                             Supabase session-pooler timeouts.
"""

import io
import logging
import os

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

CHUNK_SIZE = 100_000   # rows per commit for large-table loads


def get_connection(dsn: str | None = None):
    dsn = dsn or os.environ["DATABASE_URL"]
    if "sslmode" not in dsn:
        dsn += "?sslmode=require"
    return psycopg2.connect(dsn)


def _df_to_copy_buffer(df: pd.DataFrame) -> io.StringIO:
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    return buf


def _safe_val(v):
    """Convert NaT/NaN → None and numpy scalar types → native Python."""
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
    """INSERT … ON CONFLICT DO NOTHING via execute_values. Good for small tables."""
    records = [
        tuple(_safe_val(v) for v in row)
        for row in df[columns].itertuples(index=False)
    ]
    col_str = ", ".join(columns)
    placeholders = "(" + ", ".join(["%s"] * len(columns)) + ")"
    sql = f"INSERT INTO {table} ({col_str}) VALUES %s ON CONFLICT ({pk}) DO NOTHING"
    with conn.cursor() as cur:
        execute_values(cur, sql, records, template=placeholders, page_size=1000)
    return len(records)


def _bulk_upsert(
    conn, df: pd.DataFrame, table: str, columns: list[str], pk: str
) -> int:
    """
    COPY → temp table → INSERT ON CONFLICT DO NOTHING, committed every CHUNK_SIZE rows.

    Why temp table: COPY FROM is 10-100x faster than execute_values, but COPY doesn't
    support ON CONFLICT natively.  We COPY into a throw-away temp table, then
    INSERT-SELECT with ON CONFLICT DO NOTHING for idempotency.
    Chunked commits prevent Supabase session-pooler timeout on multi-million-row loads.
    """
    subset = df[columns].copy()
    col_str = ", ".join(columns)
    tmp = f"_tmp_{table.replace('.', '_')}"

    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TEMP TABLE IF NOT EXISTS {tmp} "
            f"AS SELECT {col_str} FROM {table} LIMIT 0"
        )
        cur.execute(f"TRUNCATE TABLE {tmp}")

    total = 0
    for start in range(0, len(subset), CHUNK_SIZE):
        chunk = subset.iloc[start : start + CHUNK_SIZE]
        buf = _df_to_copy_buffer(chunk)
        with conn.cursor() as cur:
            cur.copy_expert(
                f"COPY {tmp} ({col_str}) FROM STDIN WITH CSV NULL '\\N'", buf
            )
            cur.execute(
                f"INSERT INTO {table} ({col_str}) "
                f"SELECT {col_str} FROM {tmp} "
                f"ON CONFLICT ({pk}) DO NOTHING"
            )
            cur.execute(f"TRUNCATE TABLE {tmp}")
        conn.commit()
        total += len(chunk)
        logger.debug("  … %d / %d rows committed", total, len(subset))

    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {tmp}")

    return total


def _bulk_copy_direct(
    conn, df: pd.DataFrame, table: str, columns: list[str]
) -> int:
    """
    Direct COPY with no ON CONFLICT — use only after a TRUNCATE so there are
    no existing rows to conflict with.  Fastest possible load path.
    """
    subset = df[columns].copy()
    col_str = ", ".join(columns)
    total = 0
    for start in range(0, len(subset), CHUNK_SIZE):
        chunk = subset.iloc[start : start + CHUNK_SIZE]
        buf = _df_to_copy_buffer(chunk)
        with conn.cursor() as cur:
            cur.copy_expert(
                f"COPY {table} ({col_str}) FROM STDIN WITH CSV NULL '\\N'", buf
            )
        conn.commit()
        total += len(chunk)
        logger.debug("  … %d / %d rows copied", total, len(subset))
    return total


def truncate_all(conn) -> None:
    """
    Truncate all tables in reverse FK dependency order so every ETL re-run
    starts from a clean slate.  Without this, ON CONFLICT DO NOTHING silently
    skips rows that already exist, meaning transform fixes never reach the DB.

    Order (child → parent, so FK constraints are never violated):
      ml_encounter_features → ml_features → fact_encounters
      → dim_diagnosis / dim_lab_result / dim_medication → dim_patient
    """
    tables = [
        "ml_encounter_features",
        "ml_features",
        "fact_encounters",
        "dim_diagnosis",
        "dim_lab_result",
        "dim_medication",
        "dim_patient",
    ]
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {', '.join(tables)}")
    conn.commit()
    logger.info("Truncated all tables: %s", ", ".join(tables))


def apply_schema(conn) -> None:
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
    cols = [c for c in cols if c in df.columns]
    n = _bulk_upsert(conn, df, "fact_encounters", cols, "encounter_id")
    logger.info("Loaded %d encounters", n)
    return n


def load_diagnoses(conn, df: pd.DataFrame) -> int:
    cols = [
        "diagnosis_id", "encounter_id", "patient_id",
        "icd_code", "icd_description", "onset_date",
        "abatement_date", "clinical_status",
    ]
    cols = [c for c in cols if c in df.columns]
    n = _bulk_upsert(conn, df, "dim_diagnosis", cols, "diagnosis_id")
    logger.info("Loaded %d diagnoses", n)
    return n


def load_labs(conn, df: pd.DataFrame) -> int:
    df = df.copy()
    if "observation_date" in df.columns:
        df["observation_date"] = (
            pd.to_datetime(df["observation_date"], utc=True)
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )
    cols = [
        "lab_id", "encounter_id", "patient_id", "category", "loinc_code", "lab_name",
        "value_numeric", "value_string", "unit",
        "reference_low", "reference_high", "observation_date",
    ]
    cols = [c for c in cols if c in df.columns]
    n = _bulk_upsert(conn, df, "dim_lab_result", cols, "lab_id")
    logger.info("Loaded %d lab results", n)
    return n


def load_medications(conn, df: pd.DataFrame) -> int:
    cols = [
        "medication_id", "encounter_id", "patient_id",
        "medication_code", "medication_name",
        "start_date", "end_date", "status",
    ]
    cols = [c for c in cols if c in df.columns]
    n = _bulk_upsert(conn, df, "dim_medication", cols, "medication_id")
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


def load_encounter_features(conn, df: pd.DataFrame) -> int:
    cols = [
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
    cols = [c for c in cols if c in df.columns]

    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE ml_encounter_features")
    conn.commit()

    n = _bulk_copy_direct(conn, df, "ml_encounter_features", cols)
    logger.info("Loaded %d encounter feature rows", n)
    return n
