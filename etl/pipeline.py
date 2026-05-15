"""
ETL pipeline orchestrator.
Runs Extract → Transform → Validate → Load in sequence with structured logging.
"""

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from etl.extract import extract_all
from etl.ml_config import INPATIENT_CLASSES
from etl.load import (
    apply_schema,
    get_connection,
    load_diagnoses,
    load_encounter_features,
    load_encounters,
    load_labs,
    load_medications,
    load_ml_features,
    load_patients,
    truncate_all,
)
from etl.transform import (
    add_readmission_label,
    build_diagnoses_df,
    build_encounter_ml_features,
    build_encounters_df,
    build_labs_df,
    build_medications_df,
    build_ml_features,
    build_patients_df,
)
from etl.validate import run_all_validations

load_dotenv()

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run(csv_dir: str | Path | None = None, dsn: str | None = None) -> dict:
    """
    Run the full ETL pipeline.

    Parameters
    ----------
    csv_dir : path to Synthea CSV output directory.
              Defaults to CSV_DIR env var or 'output/csv'.
    dsn     : PostgreSQL connection string.
              Defaults to DATABASE_URL env var.

    Returns
    -------
    dict with row counts and validation summary.
    """
    t0 = time.perf_counter()

    csv_dir = Path(csv_dir or os.getenv("CSV_DIR", "output/csv"))
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    # ── Extract ───────────────────────────────────────────────────────────────
    logger.info("=== EXTRACT: reading CSV files from %s ===", csv_dir)
    t1 = time.perf_counter()
    raw = extract_all(csv_dir)
    logger.info("Extract done in %.1fs", time.perf_counter() - t1)

    # ── Transform ─────────────────────────────────────────────────────────────
    logger.info("=== TRANSFORM ===")
    t2 = time.perf_counter()

    df_patients    = build_patients_df(raw["patients"])
    df_encounters  = build_encounters_df(raw["encounters"])
    df_encounters  = add_readmission_label(df_encounters)
    df_diagnoses   = build_diagnoses_df(raw["conditions"])
    df_labs        = build_labs_df(raw["observations"])
    df_medications = build_medications_df(raw["medications"])
    df_features    = build_ml_features(
        df_patients, df_encounters, df_diagnoses, df_labs, df_medications
    )
    logger.info("Building encounter-level ML features (this may take a few minutes)...")
    df_enc_features = build_encounter_ml_features(
        df_patients, df_encounters, df_diagnoses, df_labs, df_medications
    )

    logger.info("Transform done in %.1fs", time.perf_counter() - t2)

    # ── Validate ──────────────────────────────────────────────────────────────
    logger.info("=== VALIDATE ===")
    t3 = time.perf_counter()

    report = run_all_validations(df_patients, df_encounters, df_diagnoses, df_labs, df_features, df_enc_features)
    summary = report.summary()
    logger.info(
        "Validation: %d passed, %d warnings, %d errors (%.1fs)",
        summary["passed"], summary["warnings"], summary["errors"],
        time.perf_counter() - t3,
    )
    report.raise_if_critical()

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("=== LOAD ===")
    t4 = time.perf_counter()

    # Only load inpatient/emergency encounters and their associated records.
    # Synthea generates ~703k total encounters; only ~38k are inpatient/emergency.
    # Loading all records would exhaust the Supabase 500MB free tier.
    # INPATIENT_CLASSES from ml_config is the single source of truth — it is also
    # used by add_readmission_label and build_encounter_ml_features, guaranteeing
    # that the set of encounters in the DB exactly matches what features were computed on.
    df_enc_load  = df_encounters[df_encounters["encounter_class"].isin(INPATIENT_CLASSES)].copy()
    inpatient_ids = set(df_enc_load["encounter_id"])
    df_diag_load = df_diagnoses[df_diagnoses["encounter_id"].isin(inpatient_ids)].copy()
    df_labs_load = df_labs[df_labs["encounter_id"].isin(inpatient_ids)].copy()
    df_meds_load = df_medications[df_medications["encounter_id"].isin(inpatient_ids)].copy()

    logger.info(
        "Filtered for DB load: %d encounters | %d diagnoses | %d labs | %d medications",
        len(df_enc_load), len(df_diag_load), len(df_labs_load), len(df_meds_load),
    )

    conn = get_connection(dsn)
    try:
        apply_schema(conn)
        truncate_all(conn)
        counts = {
            "patients":          load_patients(conn, df_patients),
            "encounters":        load_encounters(conn, df_enc_load),
            "diagnoses":         load_diagnoses(conn, df_diag_load),
            "labs":              load_labs(conn, df_labs_load),
            "medications":       load_medications(conn, df_meds_load),
            "ml_features":       load_ml_features(conn, df_features),
            "encounter_features": load_encounter_features(conn, df_enc_features),
        }
    finally:
        conn.close()

    logger.info("Load done in %.1fs", time.perf_counter() - t4)

    elapsed = time.perf_counter() - t0
    logger.info("=== PIPELINE COMPLETE in %.1fs ===", elapsed)
    logger.info("Row counts: %s", counts)

    return {
        "elapsed_seconds": round(elapsed, 1),
        "row_counts": counts,
        "validation": summary,
    }
