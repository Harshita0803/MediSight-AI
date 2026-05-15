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

    # Readmission rate sanity — computed on labelled encounters only (censored = NaN)
    if "readmitted_30d" in df.columns:
        labelled = df["readmitted_30d"].dropna()
        n_censored = df["readmitted_30d"].isna().sum()
        readmit_rate = labelled.mean()
        if n_censored > 0:
            report.ok(f"fact_encounters: {n_censored} censored encounters (excluded from ML)")
        if readmit_rate > 0.40:
            report.warn(f"fact_encounters: readmission rate {readmit_rate:.1%} seems high")
        else:
            report.ok(f"fact_encounters: readmission rate {readmit_rate:.1%} (on labelled encounters)")


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


def validate_encounter_features(
    df: pd.DataFrame, df_encounters: pd.DataFrame, report: ValidationReport
) -> None:
    _row_count_check(df, "ml_encounter_features", 10, report)
    _duplicate_check(df, "encounter_id", "ml_encounter_features", report)
    _referential_integrity(
        df, df_encounters, "encounter_id", "encounter_id",
        "enc_features→fact_encounters", report,
    )

    # Lab features: sum==0 catches total failure; fraction check catches partial failure
    # (e.g., only a fraction of LOINC codes normalised).  At least 30% of inpatient
    # encounters should have at least one abnormal lab — if Synthea generates realistic
    # data, virtually every encounter has some out-of-range observation.
    MIN_LAB_NONZERO_FRACTION = 0.30
    for col in ("num_abnormal_labs_this_visit", "avg_lab_deviation_this_visit"):
        if col not in df.columns:
            continue
        nonzero_frac = (df[col] > 0).mean()
        if nonzero_frac == 0:
            report.error(
                f"ml_encounter_features.{col}: all values are zero — "
                "lab normalisation likely failed entirely"
            )
        elif nonzero_frac < MIN_LAB_NONZERO_FRACTION:
            report.warn(
                f"ml_encounter_features.{col}: only {nonzero_frac:.1%} of encounters "
                f"have non-zero values (expected >= {MIN_LAB_NONZERO_FRACTION:.0%}) — "
                "partial LOINC normalisation failure?"
            )
        else:
            report.ok(f"ml_encounter_features.{col}: {nonzero_frac:.1%} encounters non-zero")

    # Chronic disease flags must have non-zero rates.
    # All-False means the flag logic in build_encounter_ml_features is broken —
    # either the current-encounter check or the prior_diag join failed silently.
    # Every flag should fire on at least some inpatient encounters in Synthea.
    for col in ("has_heart_failure", "has_diabetes", "has_copd", "has_ckd", "has_hypertension"):
        if col not in df.columns:
            continue
        n_true = df[col].sum()
        rate = df[col].mean()
        if n_true == 0:
            report.error(
                f"ml_encounter_features.{col}: all False — "
                "chronic flag logic in build_encounter_ml_features may be broken "
                "(check current_flagged | prior_flagged computation)"
            )
        else:
            report.ok(f"ml_encounter_features.{col}: {n_true} True ({rate:.1%})")

    # Readmission rate sanity check
    if "readmitted_30d" in df.columns:
        rate = df["readmitted_30d"].mean()
        if rate > 0.40:
            report.warn(f"ml_encounter_features: readmission rate {rate:.1%} seems high")
        elif rate == 0:
            report.error("ml_encounter_features: readmission rate is 0% — label generation failed")
        else:
            report.ok(f"ml_encounter_features: readmission rate {rate:.1%}")

    # No negative values in count features
    for col in ("num_labs_this_visit", "num_meds_this_visit", "num_diagnoses_this_visit"):
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                report.error(f"ml_encounter_features.{col}: {neg} negative values")
            else:
                report.ok(f"ml_encounter_features.{col}: non-negative")

    # days_since_previous_visit is nullable (NaN = first admission) but must never be negative
    if "days_since_previous_visit" in df.columns:
        neg = (df["days_since_previous_visit"].dropna() < 0).sum()
        if neg > 0:
            report.error(
                f"ml_encounter_features.days_since_previous_visit: {neg} negative values "
                "(prior_admit > admission_date — likely join ordering bug)"
            )
        else:
            report.ok("ml_encounter_features.days_since_previous_visit: non-negative")


def run_all_validations(
    df_patients: pd.DataFrame,
    df_encounters: pd.DataFrame,
    df_diagnoses: pd.DataFrame,
    df_labs: pd.DataFrame,
    df_features: pd.DataFrame,
    df_enc_features: pd.DataFrame | None = None,
) -> ValidationReport:
    report = ValidationReport()
    validate_patients(df_patients, report)
    validate_encounters(df_encounters, df_patients, report)
    validate_diagnoses(df_diagnoses, df_patients, report)
    validate_labs(df_labs, df_patients, report)
    validate_features(df_features, report)
    if df_enc_features is not None:
        validate_encounter_features(df_enc_features, df_encounters, report)
    return report
