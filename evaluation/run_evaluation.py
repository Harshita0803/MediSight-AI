import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from models.train import NUMERIC_FEATURES, BOOLEAN_FEATURES, ALL_FEATURES
from evaluation.faithfulness import FaithfulnessHarness

logger = logging.getLogger(__name__)
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True,
                    help="CSV with one row per encounter, ALL_FEATURES columns")
    ap.add_argument("--n", type=int, default=100, help="max rows to evaluate")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--llm", action="store_true",
                    help="use OpenAI explainer (needs OPENAI_API_KEY)")
    ap.add_argument("--condition", choices=["a", "b"], default="a",
                    help="a = grounded/echo control; b = reasoning (SHAP withheld)")
    ap.add_argument("--no-lime", action="store_true", help="skip SHAP-vs-LIME agreement")
    ap.add_argument("--retrieval", action="store_true",
                    help="add retrieval sanity check (needs FAISS index + patient_id column)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s")

    clf = joblib.load(MODELS_DIR / "readmission_classifier.pkl")
    df = pd.read_csv(args.features)
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = float("nan")
    patient_ids = df["patient_id"].astype(str).tolist() if "patient_id" in df.columns else None
    df = df.head(args.n).reset_index(drop=True)
    if patient_ids:
        patient_ids = patient_ids[: args.n]

    logger.info("Evaluating %d encounters (top_k=%d)", len(df), args.top_k)

    explanation_fn = None
    if args.llm:
        from evaluation.explain import make_explainer
        explanation_fn = make_explainer(condition=args.condition.upper())
        logger.info("Using OpenAI explainer — Condition %s", args.condition.upper())
    else:
        logger.info("Using deterministic template explainer (offline mode)")

    retriever_factory = None
    if args.retrieval:
        from rag.retriever import load_retriever
        retriever_factory = load_retriever
        logger.info("Retrieval sanity check enabled")

    harness = FaithfulnessHarness(
        clf, NUMERIC_FEATURES, BOOLEAN_FEATURES,
        explanation_fn=explanation_fn,
        retriever_factory=retriever_factory,
        top_k=args.top_k,
    )

    out = harness.evaluate_cohort(df, patient_ids=patient_ids, with_lime=not args.no_lime)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"cond{args.condition.upper()}" if args.llm else "template"
    out_path = REPORTS_DIR / f"faithfulness_results_{tag}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))

    print("\n=== Cohort summary ===")
    print(json.dumps(out["summary"], indent=2, default=str))
    print(f"\nFull results -> {out_path}")


if __name__ == "__main__":
    main()
