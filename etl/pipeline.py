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
from etl.load import (
    apply_schema,
    get_connection,
    load_diagnoses,
    load_encounters,
    load_labs,
    load_medications,
    load_ml_features,
    load_patients,
)
from etl.transform import (
    add_readmission_label,
    build_diagnoses_df,
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

    logger.info("Transform done in %.1fs", time.perf_counter() - t2)

    # ── Validate ──────────────────────────────────────────────────────────────
    logger.info("=== VALIDATE ===")
    t3 = time.perf_counter()

    report = run_all_validations(df_patients, df_encounters, df_diagnoses, df_labs, df_features)
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

    conn = get_connection(dsn)
    try:
        apply_schema(conn)
        counts = {
            "patients":    load_patients(conn, df_patients),
            "encounters":  load_encounters(conn, df_encounters),
            "diagnoses":   load_diagnoses(conn, df_diagnoses),
            "labs":        load_labs(conn, df_labs),
            "medications": load_medications(conn, df_medications),
            "ml_features": load_ml_features(conn, df_features),
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
