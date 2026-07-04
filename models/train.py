import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold, RandomizedSearchCV, StratifiedGroupKFold,
    cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import sqlalchemy

from etl.ml_config import SPLIT_RANDOM_STATE, SPLIT_TEST_SIZE

load_dotenv()
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent
MLFLOW_EXPERIMENT = "medisight-readmission"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def _db_engine():
    db_url = os.environ["DATABASE_URL"].replace("postgres://", "postgresql://")
    if "sslmode" not in db_url:
        db_url += "?sslmode=require"
    return sqlalchemy.create_engine(db_url)


def load_data() -> pd.DataFrame:
    engine = _db_engine()
    logger.info("Loading ml_encounter_features from database...")
    df = pd.read_sql("SELECT * FROM ml_encounter_features ORDER BY encounter_id", engine)
    engine.dispose()
    logger.info(
        "Loaded %d encounters — %.1f%% readmitted",
        len(df), df["readmitted_30d"].mean() * 100,
    )
    return df


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    null_labels = df["readmitted_30d"].isna().sum()
    if null_labels > 0:
        raise ValueError(
            f"{null_labels} NULL values in readmitted_30d. "
            "Censored encounters should have been dropped by build_encounter_ml_features(). "
            "Re-run the ETL pipeline to rebuild the table."
        )
    df["readmitted_30d"] = df["readmitted_30d"].astype(bool)
    return df


NUMERIC_FEATURES = [
    "age_at_admission", "gender_encoded", "insurance_risk_tier",
    "encounter_class_encoded", "length_of_stay_days",
    "num_diagnoses_this_visit",
    "num_labs_this_visit", "num_abnormal_labs_this_visit", "avg_lab_deviation_this_visit",
    "num_meds_this_visit",
    "prior_admissions_6m", "prior_admissions_12m", "prior_admissions_total",
    "days_since_previous_visit", "comorbidity_count_prior",
]
BOOLEAN_FEATURES = [
    "has_heart_failure", "has_diabetes", "has_copd", "has_ckd", "has_hypertension",
    "is_first_admission",
]

ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES


def build_preprocessor(
    numeric_features: list[str] | None = None,
    boolean_features: list[str] | None = None,
) -> ColumnTransformer:
    num = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    bools = boolean_features if boolean_features is not None else BOOLEAN_FEATURES
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, num),
        ("bool", "passthrough", bools),
    ])


def _patient_split(df: pd.DataFrame, test_size: float = SPLIT_TEST_SIZE,
                   random_state: int = SPLIT_RANDOM_STATE):
    patient_labels = (
        df.groupby("patient_id")["readmitted_30d"]
        .any()
        .astype(int)
        .reset_index()
        .rename(columns={"readmitted_30d": "has_any_readmit"})
    )
    train_pats, test_pats = train_test_split(
        patient_labels["patient_id"],
        test_size=test_size,
        stratify=patient_labels["has_any_readmit"],
        random_state=random_state,
    )
    train_mask = df["patient_id"].isin(set(train_pats))
    test_mask = df["patient_id"].isin(set(test_pats))
    return train_mask, test_mask


def train_readmission_classifier(df: pd.DataFrame) -> dict:
    logger.info("=== Training readmission classifier ===")

    features = [f for f in ALL_FEATURES if f in df.columns]
    X = df[features].copy()
    y = df["readmitted_30d"].astype(int)

    num_features = [f for f in NUMERIC_FEATURES if f in df.columns]
    bool_features = [f for f in BOOLEAN_FEATURES if f in df.columns]
    for col in num_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in bool_features:
        X[col] = X[col].fillna(False).astype(int)

    logger.info(
        "Class distribution — positive: %d (%.1f%%), negative: %d",
        y.sum(), y.mean() * 100, (y == 0).sum(),
    )

    train_mask, test_mask = _patient_split(df)
    X_train, X_test = X[train_mask.values], X[test_mask.values]
    y_train, y_test = y[train_mask.values], y[test_mask.values]
    groups_train = df["patient_id"][train_mask].values
    logger.info(
        "Train: %d encounters (%d patients) | Test: %d encounters (%d patients)",
        len(X_train), len(np.unique(groups_train)),
        len(X_test), df["patient_id"][test_mask].nunique(),
    )

    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())

    preprocessor = build_preprocessor(num_features, bool_features)
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            scale_pos_weight=neg / max(pos, 1),
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )),
    ])

    param_grid = {
        "clf__n_estimators":     [100, 200, 300],
        "clf__max_depth":        [3, 4, 5, 6],
        "clf__learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "clf__subsample":        [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5],
    }

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"xgboost-readmission-{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}"):
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=20, cv=cv,
            scoring="roc_auc", random_state=42, n_jobs=-1,
        )
        search.fit(X_train, y_train, groups=groups_train)
        best = search.best_estimator_

        oof_probs = cross_val_predict(
            best, X_train, y_train, cv=cv, method="predict_proba", groups=groups_train
        )[:, 1]
        thresholds = np.linspace(0.05, 0.95, 181)
        f1_scores_thr = [f1_score(y_train.values, oof_probs >= t) for t in thresholds]
        optimal_threshold = float(thresholds[int(np.argmax(f1_scores_thr))])
        logger.info(
            "Threshold tuning: optimal=%.3f  OOF-F1=%.4f",
            optimal_threshold, max(f1_scores_thr),
        )

        y_prob = best.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= optimal_threshold).astype(int)

        metrics = {
            "test_roc_auc":      round(roc_auc_score(y_test, y_prob), 4),
            "test_pr_auc":       round(average_precision_score(y_test, y_prob), 4),
            "test_brier":        round(brier_score_loss(y_test, y_prob), 4),
            "test_f1":           round(f1_score(y_test, y_pred), 4),
            "test_precision":    round(precision_score(y_test, y_pred, zero_division=0), 4),
            "test_recall":       round(recall_score(y_test, y_pred), 4),
            "cv_auc":            round(search.best_score_, 4),
            "optimal_threshold": round(optimal_threshold, 4),
        }

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best, "readmission_classifier")

        logger.info(
            "Readmission — ROC-AUC: %.4f | PR-AUC: %.4f | Brier: %.4f | "
            "F1: %.4f | Precision: %.4f | Recall: %.4f | CV-AUC: %.4f",
            metrics["test_roc_auc"], metrics["test_pr_auc"], metrics["test_brier"],
            metrics["test_f1"], metrics["test_precision"],
            metrics["test_recall"], metrics["cv_auc"],
        )

    out_path = MODELS_DIR / "readmission_classifier.pkl"
    joblib.dump(best, out_path)
    logger.info("Saved to %s", out_path)

    return {
        "model": best, "metrics": metrics,
        "X_train": X_train, "X_test": X_test, "y_test": y_test, "features": features,
    }


def train_los_regressor(df: pd.DataFrame) -> dict:
    logger.info("=== Training length-of-stay regressor ===")

    full_train_mask, full_test_mask = _patient_split(df)
    los_filter = df["length_of_stay_days"] > 0
    los_train_mask = full_train_mask & los_filter
    los_test_mask = full_test_mask & los_filter

    n_los_train_patients = df.loc[los_train_mask, "patient_id"].nunique()
    logger.info(
        "LOS training set: %d encounters (%d patients) | test: %d encounters",
        los_train_mask.sum(), n_los_train_patients, los_test_mask.sum(),
    )
    if n_los_train_patients < 10:
        raise ValueError(
            f"LOS training set has only {n_los_train_patients} patients — "
            "too few for GroupKFold(n_splits=5). Check that length_of_stay_days is "
            "populated and the ETL loaded inpatient encounters correctly."
        )

    los_features = [f for f in ALL_FEATURES if f != "length_of_stay_days" and f in df.columns]
    num_features = [f for f in NUMERIC_FEATURES if f != "length_of_stay_days" and f in df.columns]

    X_train = df.loc[los_train_mask, los_features].copy()
    X_test = df.loc[los_test_mask, los_features].copy()
    y_train = np.log1p(df.loc[los_train_mask, "length_of_stay_days"])
    y_test_days = df.loc[los_test_mask, "length_of_stay_days"]
    groups_train_los = df.loc[los_train_mask, "patient_id"].values

    for col in num_features:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
    for col in BOOLEAN_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna(False).astype(int)
            X_test[col] = X_test[col].fillna(False).astype(int)

    logger.info("Train: %d rows (%d patients) | Test: %d rows",
                len(X_train), len(np.unique(groups_train_los)), len(X_test))

    bool_features_present = [f for f in BOOLEAN_FEATURES if f in X_train.columns]
    los_preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_features),
        ("bool", "passthrough", bool_features_present),
    ])
    pipeline = Pipeline([
        ("prep", los_preprocessor),
        ("reg", GradientBoostingRegressor(random_state=42)),
    ])

    param_grid = {
        "reg__n_estimators":     [100, 200, 300],
        "reg__max_depth":        [3, 4, 5],
        "reg__learning_rate":    [0.05, 0.1, 0.2],
        "reg__subsample":        [0.7, 0.8, 1.0],
        "reg__min_samples_leaf": [1, 3, 5],
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"gbm-los-regressor-logtransform-{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}"):
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=15,
            cv=GroupKFold(n_splits=5),
            scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
        )
        search.fit(X_train, y_train, groups=groups_train_los)
        best = search.best_estimator_

        y_pred = np.expm1(best.predict(X_test))
        metrics = {
            "test_mae":   round(mean_absolute_error(y_test_days, y_pred), 4),
            "test_rmse":  round(np.sqrt(mean_squared_error(y_test_days, y_pred)), 4),
            "test_r2":    round(r2_score(y_test_days, y_pred), 4),
            "cv_mae_log": round(-search.best_score_, 4),
        }

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best, "los_regressor")

        logger.info(
            "LOS — MAE: %.4f days | RMSE: %.4f | R²: %.4f | CV-MAE (log): %.4f",
            metrics["test_mae"], metrics["test_rmse"], metrics["test_r2"], metrics["cv_mae_log"],
        )

    out_path = MODELS_DIR / "los_regressor.pkl"
    joblib.dump(best, out_path)
    logger.info("Saved to %s", out_path)

    return {"model": best, "metrics": metrics}


def compute_shap(clf_result: dict) -> dict:
    logger.info("=== Computing SHAP values ===")

    model = clf_result["model"]
    X_test = clf_result["X_test"]
    y_test = clf_result["y_test"]
    features = clf_result["features"]

    prep = model.named_steps["prep"]
    xgb_model = model.named_steps["clf"]

    sample_size = min(2000, len(X_test))
    sample_idx = X_test.sample(n=sample_size, random_state=42).index
    X_sample = X_test.loc[sample_idx]
    y_sample = y_test.loc[sample_idx].values
    X_transformed = prep.transform(X_sample)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed)

    num_names = [f for f in NUMERIC_FEATURES if f in features]
    bool_names = [f for f in BOOLEAN_FEATURES if f in features]
    feature_names = num_names + bool_names

    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names,
    ).sort_values(ascending=False)

    logger.info("Top 5 features by SHAP importance:")
    for feat, val in mean_shap.head(5).items():
        logger.info("  %-40s %.4f", feat, val)

    top_in_sample = [f for f in list(mean_shap.head(10).index) if f in X_sample.columns]
    if len(top_in_sample) >= 2:
        corr_matrix = X_sample[top_in_sample].corr(method="spearman")
        high_corr: list[tuple] = []
        for i, fa in enumerate(top_in_sample):
            for fb in top_in_sample[i + 1:]:
                r = corr_matrix.loc[fa, fb]
                if abs(r) > 0.5:
                    high_corr.append((fa, fb, r))
        if high_corr:
            logger.warning("High-correlation pairs among top-10 SHAP features:")
            for fa, fb, r in high_corr:
                logger.warning("  %s  ↔  %s   r=%.3f", fa, fb, r)

    UTILIZATION_GROUP = [
        "prior_admissions_6m", "prior_admissions_12m",
        "prior_admissions_total", "days_since_previous_visit",
    ]
    group_indices = [feature_names.index(f) for f in UTILIZATION_GROUP if f in feature_names]
    group_delta_auc: float | None = None
    if len(group_indices) >= 2:
        baseline_auc = roc_auc_score(y_sample, xgb_model.predict_proba(X_transformed)[:, 1])
        rng = np.random.default_rng(42)
        perm_aucs = []
        for _ in range(30):
            X_perm = X_transformed.copy()
            perm_order = rng.permutation(len(X_perm))
            X_perm[:, group_indices] = X_perm[perm_order][:, group_indices]
            perm_aucs.append(roc_auc_score(y_sample, xgb_model.predict_proba(X_perm)[:, 1]))
        group_delta_auc = baseline_auc - float(np.mean(perm_aucs))
        logger.info(
            "Group permutation — utilization history: baseline AUC=%.4f  ΔAUC=%.4f",
            baseline_auc, group_delta_auc,
        )

    shap_path = MODELS_DIR / "shap_values.pkl"
    joblib.dump({
        "shap_values":               shap_values,
        "feature_names":             feature_names,
        "expected_value":            explainer.expected_value,
        "X_transformed":             X_transformed,
        "mean_importance":           mean_shap.to_dict(),
        "utilization_group_delta_auc": group_delta_auc,
    }, shap_path)
    logger.info("SHAP values saved to %s", shap_path)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"shap-artifacts-{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}"):
        summary_path = REPORTS_DIR / "shap_summary.png"
        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(str(summary_path), dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(summary_path))

        waterfall_path = REPORTS_DIR / "shap_waterfall_example.png"
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=float(explainer.expected_value),
                data=X_transformed[0],
                feature_names=feature_names,
            ),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(str(waterfall_path), dpi=150, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(waterfall_path))

        for feat, val in mean_shap.head(10).items():
            mlflow.log_metric(f"shap_{feat}", round(float(val), 6))

    return {
        "shap_values":     shap_values,
        "feature_names":   feature_names,
        "mean_importance": mean_shap,
        "group_delta_auc": group_delta_auc,
    }


def run_statistical_analysis(df: pd.DataFrame) -> None:
    logger.info("=== Statistical analysis ===")

    for tier in sorted(df["insurance_risk_tier"].dropna().unique()):
        rate = df[df["insurance_risk_tier"] == tier]["readmitted_30d"].mean()
        logger.info("  Insurance tier %d readmission rate: %.1f%%", int(tier), rate * 100)

    for flag in ["has_heart_failure", "has_diabetes", "has_copd", "has_ckd"]:
        if flag in df.columns:
            rate_yes = df[df[flag] == True]["readmitted_30d"].mean()
            rate_no = df[df[flag] == False]["readmitted_30d"].mean()
            logger.info("  %-20s with: %.1f%% | without: %.1f%%", flag, rate_yes * 100, rate_no * 100)

    if "is_first_admission" in df.columns:
        rate_first = df[df["is_first_admission"] == 1]["readmitted_30d"].mean()
        rate_returning = df[df["is_first_admission"] == 0]["readmitted_30d"].mean()
        logger.info(
            "  First-time admissions: %.1f%% | Returning patients: %.1f%%",
            rate_first * 100, rate_returning * 100,
        )


def run_training() -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    df = load_data()
    df = build_dataset(df)
    run_statistical_analysis(df)

    clf_result = train_readmission_classifier(df)
    los_result = train_los_regressor(df)
    shap_result = compute_shap(clf_result)

    logger.info("=== TRAINING COMPLETE ===")
    logger.info(
        "Readmission — ROC-AUC: %.4f | CV-AUC: %.4f | gap: %+.4f | "
        "PR-AUC: %.4f | Brier: %.4f | F1: %.4f (thr=%.3f) | Recall: %.4f",
        clf_result["metrics"]["test_roc_auc"], clf_result["metrics"]["cv_auc"],
        clf_result["metrics"]["test_roc_auc"] - clf_result["metrics"]["cv_auc"],
        clf_result["metrics"]["test_pr_auc"], clf_result["metrics"]["test_brier"],
        clf_result["metrics"]["test_f1"], clf_result["metrics"]["optimal_threshold"],
        clf_result["metrics"]["test_recall"],
    )
    logger.info(
        "LOS — MAE: %.4f days | R²: %.4f",
        los_result["metrics"]["test_mae"], los_result["metrics"]["test_r2"],
    )
    logger.info("Top feature — %s", list(shap_result["mean_importance"].keys())[0])

    return {
        "readmission": clf_result["metrics"],
        "los": los_result["metrics"],
        "top_features": dict(list(shap_result["mean_importance"].items())[:5]),
    }


if __name__ == "__main__":
    results = run_training()
    gap = round(results['readmission']['test_roc_auc'] - results['readmission']['cv_auc'], 4)
    print("\n=== RESULTS ===")
    print(f"Readmission ROC-AUC       : {results['readmission']['test_roc_auc']}")
    print(f"Readmission CV-AUC        : {results['readmission']['cv_auc']}  (overfit gap = {gap:+.4f})")
    print(f"Readmission PR-AUC        : {results['readmission']['test_pr_auc']}")
    print(f"Readmission Brier         : {results['readmission']['test_brier']}")
    print(f"Readmission F1            : {results['readmission']['test_f1']}  (threshold={results['readmission']['optimal_threshold']})")
    print(f"Readmission Precision     : {results['readmission']['test_precision']}")
    print(f"Readmission Recall        : {results['readmission']['test_recall']}")
    print(f"LOS MAE                   : {results['los']['test_mae']} days")
    print(f"LOS R²                    : {results['los']['test_r2']}")
    print(f"Top features: {list(results['top_features'].keys())}")
