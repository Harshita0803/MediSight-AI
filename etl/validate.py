"""
Data quality checks run after transformation, before loading.
Returns a summary dict and raises ValueError on critical failures.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

CRITICAL_NULL_THRESHOLD = 0.50   # fail if >50 % nulls in a critical column
WARN_NULL_THRESHOLD = 0.20


@dataclass
class ValidationReport:
    passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def ok(self, msg: str) -> None:
        self.passed.append(msg)
        logger.debug("[PASS] %s", msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning("[WARN] %s", msg)

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        logger.error("[FAIL] %s", msg)

    def raise_if_critical(self) -> None:
        if self.errors:
            raise ValueError("Critical data quality failures:\n" + "\n".join(self.errors))

    def summary(self) -> dict:
        return {
            "passed": len(self.passed),
            "warnings": len(self.warnings),
            "errors": len(self.errors),
            "details": {
                "passed": self.passed,
                "warnings": self.warnings,
                "errors": self.errors,
            },
        }


def _null_rate_check(
    df: pd.DataFrame,
    col: str,
    label: str,
    report: ValidationReport,
    critical: bool = False,
) -> None:
    rate = df[col].isna().mean()
    if rate > CRITICAL_NULL_THRESHOLD and critical:
        report.error(f"{label}.{col}: {rate:.1%} nulls (>{CRITICAL_NULL_THRESHOLD:.0%})")
    elif rate > WARN_NULL_THRESHOLD:
        report.warn(f"{label}.{col}: {rate:.1%} nulls")
    else:
        report.ok(f"{label}.{col}: {rate:.1%} nulls")


def _duplicate_check(df: pd.DataFrame, key_col: str, label: str, report: ValidationReport) -> None:
    dupes = df[key_col].duplicated().sum()
    if dupes > 0:
        report.error(f"{label}: {dupes} duplicate {key_col} values")
    else:
        report.ok(f"{label}: no duplicate {key_col}")


def _row_count_check(df: pd.DataFrame, label: str, min_rows: int, report: ValidationReport) -> None:
    if len(df) < min_rows:
        report.error(f"{label}: only {len(df)} rows (expected >= {min_rows})")
    else:
        report.ok(f"{label}: {len(df)} rows")


def _referential_integrity(
    child_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    child_col: str,
    parent_col: str,
    label: str,
    report: ValidationReport,
) -> None:
    orphans = ~child_df[child_col].isin(parent_df[parent_col])
    orphan_count = orphans.sum()
    rate = orphans.mean()
    if rate > 0.10:
        report.error(f"{label}: {orphan_count} ({rate:.1%}) orphan {child_col} values")
    elif orphan_count > 0:
        report.warn(f"{label}: {orphan_count} ({rate:.1%}) orphan {child_col} values")
    else:
        report.ok(f"{label}: referential integrity OK")


def validate_patients(df: pd.DataFrame, report: ValidationReport) -> None:
    _row_count_check(df, "dim_patient", 10, report)
    _duplicate_check(df, "patient_id", "dim_patient", report)
    for col in ("patient_id", "gender"):
        _null_rate_check(df, col, "dim_patient", report, critical=True)
    _null_rate_check(df, "birth_date", "dim_patient", report, critical=False)
    _null_rate_check(df, "insurance_type", "dim_patient", report)


def validate_encounters(df: pd.DataFrame, df_patients: pd.DataFrame, report: ValidationReport) -> None:
    _row_count_check(df, "fact_encounters", 10, report)
    _duplicate_check(df, "encounter_id", "fact_encounters", report)
    _null_rate_check(df, "admission_date", "fact_encounters", report, critical=True)
    _null_rate_check(df, "patient_id", "fact_encounters", report, critical=True)
    _referential_integrity(df, df_patients, "patient_id", "patient_id", "encounters→patients", report)

    # LOS range check
    los = df["length_of_stay_days"].dropna()
    neg = (los < 0).sum()
    extreme = (los > 180).sum()
    if neg > 0:
        report.error(f"fact_encounters: {neg} encounters with negative LOS")
    else:
        report.ok("fact_encounters: LOS values non-negative")
    if extreme > 0:
        report.warn(f"fact_encounters: {extreme} encounters with LOS > 180 days")

    # Readmission rate sanity
    readmit_rate = df["readmitted_30d"].mean() if "readmitted_30d" in df.columns else None
    if readmit_rate is not None:
        if readmit_rate > 0.40:
            report.warn(f"fact_encounters: readmission rate {readmit_rate:.1%} seems high")
        else:
            report.ok(f"fact_encounters: readmission rate {readmit_rate:.1%}")


def validate_diagnoses(df: pd.DataFrame, df_patients: pd.DataFrame, report: ValidationReport) -> None:
    _row_count_check(df, "dim_diagnosis", 10, report)
    _duplicate_check(df, "diagnosis_id", "dim_diagnosis", report)
    _null_rate_check(df, "icd_code", "dim_diagnosis", report, critical=True)
    _referential_integrity(df, df_patients, "patient_id", "patient_id", "diagnoses→patients", report)


def validate_labs(df: pd.DataFrame, df_patients: pd.DataFrame, report: ValidationReport) -> None:
    _duplicate_check(df, "lab_id", "dim_lab_result", report)
    _null_rate_check(df, "value_numeric", "dim_lab_result", report)
    _referential_integrity(df, df_patients, "patient_id", "patient_id", "labs→patients", report)


def validate_features(df: pd.DataFrame, report: ValidationReport) -> None:
    _row_count_check(df, "ml_features", 1, report)
    _duplicate_check(df, "patient_id", "ml_features", report)
    for col in ("age", "comorbidity_count", "medication_count"):
        if col in df.columns:
            negatives = (df[col] < 0).sum()
            if negatives > 0:
                report.error(f"ml_features.{col}: {negatives} negative values")
            else:
                report.ok(f"ml_features.{col}: values non-negative")


def run_all_validations(
    df_patients: pd.DataFrame,
    df_encounters: pd.DataFrame,
    df_diagnoses: pd.DataFrame,
    df_labs: pd.DataFrame,
    df_features: pd.DataFrame,
) -> ValidationReport:
    report = ValidationReport()
    validate_patients(df_patients, report)
    validate_encounters(df_encounters, df_patients, report)
    validate_diagnoses(df_diagnoses, df_patients, report)
    validate_labs(df_labs, df_patients, report)
    validate_features(df_features, report)
    return report
