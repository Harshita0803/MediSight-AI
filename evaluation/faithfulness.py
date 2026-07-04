from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau

from evaluation.lexicon import SYNONYMS, INCREASE_CUES, DECREASE_CUES

logger = logging.getLogger(__name__)

INCREASE, DECREASE, UNCLEAR = "increase", "decrease", "unclear"

_PHRASE_RX: dict[str, "re.Pattern"] = {}


def _rx(phrase: str) -> "re.Pattern":
    if phrase not in _PHRASE_RX:
        _PHRASE_RX[phrase] = re.compile(r"\b" + re.escape(phrase) + r"\b")
    return _PHRASE_RX[phrase]


_FEATURE_PHRASES: list[tuple[str, str]] = sorted(
    ((feat, p) for feat, ps in SYNONYMS.items() for p in ps),
    key=lambda fp: len(fp[1]), reverse=True,
)


def _find_mentions(text: str) -> dict[str, int]:
    low = text.lower()
    claimed = [False] * len(low)
    found: dict[str, int] = {}
    for feat, phrase in _FEATURE_PHRASES:
        for m in _rx(phrase).finditer(low):
            s, e = m.span()
            if any(claimed[s:e]):
                continue
            for i in range(s, e):
                claimed[i] = True
            if feat not in found:
                found[feat] = s
    return found


_SENT_SPLIT = re.compile(r"[.!?;]+")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    starts = [0] + [m.end() for m in _SENT_SPLIT.finditer(text)]
    ends = [m.start() for m in _SENT_SPLIT.finditer(text)] + [len(text)]
    return list(zip(starts, ends))


def _cues(window: str) -> tuple[bool, bool]:
    return (any(c in window for c in INCREASE_CUES),
            any(c in window for c in DECREASE_CUES))


def _direction_in_context(text: str, pos: int) -> str:
    low = text.lower()
    spans = _sentence_spans(low)
    idx = next((i for i, (s, e) in enumerate(spans) if s <= pos <= e), None)
    if idx is None:
        return UNCLEAR

    s, e = spans[idx]
    inc, dec = _cues(low[s:e])
    if inc and not dec:
        return INCREASE
    if dec and not inc:
        return DECREASE
    if inc and dec:
        return UNCLEAR

    for j in range(idx - 1, -1, -1):
        pinc, pdec = _cues(low[spans[j][0]:spans[j][1]])
        if pinc and not pdec:
            return INCREASE
        if pdec and not pinc:
            return DECREASE
        if pinc and pdec:
            break
    return UNCLEAR


def classify_feature(text: str, feature: str) -> tuple[int, str]:
    mentions = _find_mentions(text)
    if feature not in mentions:
        return 0, ""
    return 1, _direction_in_context(text, mentions[feature])


@dataclass
class PatientReport:
    patient_id: Optional[str]
    risk_prob: float
    shap_top: list[tuple[str, float]]
    explanation_text: str
    parsed_claims: dict[str, str]
    nl_faithfulness: dict[str, float] = field(default_factory=dict)
    model_grounding: dict[str, float] = field(default_factory=dict)
    lime_agreement: dict[str, float] = field(default_factory=dict)
    retrieval_grounding: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        return asdict(self)


class FaithfulnessHarness:
    def __init__(
        self,
        clf_pipeline,
        numeric_features: list[str],
        boolean_features: list[str],
        explanation_fn: Optional[Callable[..., str]] = None,
        retriever_factory: Optional[Callable[[str], Any]] = None,
        top_k: int = 5,
    ):
        self.pipeline = clf_pipeline
        self.prep = clf_pipeline.named_steps["prep"]
        self.model = clf_pipeline.named_steps["clf"]
        self.numeric_features = list(numeric_features)
        self.boolean_features = list(boolean_features)
        self.feature_names = self.numeric_features + self.boolean_features
        self.explanation_fn = explanation_fn or self._template_explanation
        self.retriever_factory = retriever_factory
        self.top_k = top_k
        self._explainer = shap.TreeExplainer(self.model)
        self._lime = None

    def _shap_row(self, x_row_t: np.ndarray) -> np.ndarray:
        sv = self._explainer.shap_values(x_row_t.reshape(1, -1))
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else sv[0]
        return np.asarray(sv)[0]

    def _top_features(self, shap_vec: np.ndarray) -> list[tuple[str, float]]:
        order = np.argsort(np.abs(shap_vec))[::-1][: self.top_k]
        return [(self.feature_names[i], float(shap_vec[i])) for i in order]

    @staticmethod
    def _template_explanation(shap_top: list[tuple[str, float]], risk_prob: float,
                              feature_values: dict | None = None) -> str:
        parts = [f"The model estimates a readmission probability of {risk_prob:.0%}."]
        for feat, val in shap_top:
            direction = "increases" if val > 0 else "decreases"
            parts.append(f"The feature {feat.replace('_', ' ')} {direction} the predicted risk.")
        return " ".join(parts)

    def _parse_claims(self, text: str, candidate_features: list[str]) -> dict[str, str]:
        claims: dict[str, str] = {}
        for feat in candidate_features:
            mentioned, direction = classify_feature(text, feat)
            if mentioned:
                claims[feat] = direction
        return claims

    def _all_mentioned(self, text: str) -> set[str]:
        return set(_find_mentions(text))

    def _nl_faithfulness(
        self, shap_top: list[tuple[str, float]], text: str
    ) -> tuple[dict[str, float], dict[str, str]]:
        top_feats = [f for f, _ in shap_top]
        top_dir = {f: (INCREASE if v > 0 else DECREASE) for f, v in shap_top}

        claims = self._parse_claims(text, top_feats)
        mentioned_top = set(claims)
        mentioned_all = self._all_mentioned(text)
        fabricated = mentioned_all - set(top_feats)

        coverage = len(mentioned_top) / max(len(top_feats), 1)
        fabrication = len(fabricated) / max(len(mentioned_all), 1)

        directed = [f for f in mentioned_top if claims[f] != UNCLEAR]
        correct = sum(1 for f in directed if claims[f] == top_dir[f])
        wrong = len(directed) - correct
        n_unclear = len(mentioned_top) - len(directed)

        direction_accuracy = correct / max(len(directed), 1) if directed else float("nan")
        direction_error = wrong / max(len(directed), 1) if directed else float("nan")
        direction_unclear = n_unclear / max(len(mentioned_top), 1) if mentioned_top else float("nan")

        rank_fidelity = self._rank_fidelity(shap_top, text, mentioned_top)

        def _r(v):
            return round(v, 4) if v == v else float("nan")

        return {
            "coverage":           round(coverage, 4),
            "fabrication":        round(fabrication, 4),
            "direction_accuracy": _r(direction_accuracy),
            "direction_error":    _r(direction_error),
            "direction_unclear":  _r(direction_unclear),
            "rank_fidelity":      _r(rank_fidelity),
        }, claims

    def _rank_fidelity(self, shap_top, text, mentioned_top) -> float:
        feats = [f for f, _ in shap_top if f in mentioned_top]
        if len(feats) < 2:
            return float("nan")
        pos = _find_mentions(text)
        shap_rank = list(range(len(feats)))
        text_rank = list(np.argsort([pos.get(f, 10 ** 9) for f in feats]))
        tau, _ = kendalltau(shap_rank, text_rank)
        return float(tau) if tau == tau else float("nan")

    def _model_grounding(self, x_row_t: np.ndarray, shap_top) -> dict[str, float]:
        idx = [self.feature_names.index(f) for f, _ in shap_top]
        full = float(self.model.predict_proba(x_row_t.reshape(1, -1))[0, 1])

        masked = x_row_t.copy()
        masked[idx] = 0.0
        comp = full - float(self.model.predict_proba(masked.reshape(1, -1))[0, 1])

        only = np.zeros_like(x_row_t)
        only[idx] = x_row_t[idx]
        suff = full - float(self.model.predict_proba(only.reshape(1, -1))[0, 1])

        return {
            "prediction":        round(full, 4),
            "comprehensiveness": round(comp, 4),
            "sufficiency":       round(suff, 4),
        }

    def _fit_lime(self, X_t: np.ndarray):
        from lime.lime_tabular import LimeTabularExplainer
        self._lime = LimeTabularExplainer(
            training_data=X_t,
            feature_names=self.feature_names,
            class_names=["no_readmit", "readmit"],
            discretize_continuous=True,
            mode="classification",
            random_state=42,
        )

    def _lime_agreement(self, x_row_t: np.ndarray, shap_top) -> dict[str, float]:
        if self._lime is None:
            return {}
        exp = self._lime.explain_instance(
            x_row_t, self.model.predict_proba,
            num_features=self.top_k, num_samples=1000,
        )
        lime_pairs = {self.feature_names[i]: w for i, w in exp.as_map()[1]}
        lime_top = list(lime_pairs)
        shap_feats = [f for f, _ in shap_top]
        shap_signs = {f: (1 if v > 0 else -1) for f, v in shap_top}

        inter = set(shap_feats) & set(lime_top)
        union = set(shap_feats) | set(lime_top)
        jaccard = len(inter) / max(len(union), 1)
        sign_agree = (
            sum(1 for f in inter if shap_signs[f] == (1 if lime_pairs[f] > 0 else -1))
            / max(len(inter), 1)
        ) if inter else float("nan")

        if len(inter) >= 2:
            sr = [shap_feats.index(f) for f in inter]
            lr = [lime_top.index(f) for f in inter]
            tau, _ = kendalltau(sr, lr)
            rank_tau = float(tau) if tau == tau else float("nan")
        else:
            rank_tau = float("nan")

        return {
            "jaccard":        round(jaccard, 4),
            "sign_agreement": round(sign_agree, 4) if sign_agree == sign_agree else float("nan"),
            "rank_tau":       round(rank_tau, 4) if rank_tau == rank_tau else float("nan"),
        }

    def _retrieval_grounding(self, patient_id: str, shap_top) -> dict[str, Any]:
        retriever = self.retriever_factory(patient_id)
        per_feature: dict[str, bool] = {}
        for feat, _ in shap_top:
            phrases = SYNONYMS.get(feat, [feat])
            docs = retriever.invoke(phrases[0])
            blob = " ".join(getattr(d, "page_content", "") for d in docs).lower()
            per_feature[feat] = any(p in blob for p in phrases)
        rate = sum(per_feature.values()) / max(len(per_feature), 1)
        return {
            "per_feature_supported": per_feature,
            "grounding_rate": round(rate, 4),
            "caveat": (
                "Synthea notes are generated from the structured record; "
                "high grounding is expected by construction (instrument sanity check)."
            ),
        }

    def evaluate_patient(self, row_df: pd.DataFrame, patient_id: Optional[str] = None) -> PatientReport:
        x_t = self.prep.transform(row_df[self.feature_names])[0]
        shap_vec = self._shap_row(x_t)
        shap_top = self._top_features(shap_vec)
        risk = float(self.model.predict_proba(x_t.reshape(1, -1))[0, 1])

        feature_values = {
            f: (None if pd.isna(v) else float(v))
            for f, v in row_df[self.feature_names].iloc[0].items()
        }

        text = self.explanation_fn(shap_top, risk, feature_values)
        nl, claims = self._nl_faithfulness(shap_top, text)
        grounding = self._model_grounding(x_t, shap_top)
        lime = self._lime_agreement(x_t, shap_top)
        retr = (
            self._retrieval_grounding(patient_id, shap_top)
            if (self.retriever_factory and patient_id) else None
        )

        return PatientReport(
            patient_id=patient_id, risk_prob=round(risk, 4), shap_top=shap_top,
            explanation_text=text, parsed_claims=claims, nl_faithfulness=nl,
            model_grounding=grounding, lime_agreement=lime, retrieval_grounding=retr,
        )

    def evaluate_cohort(
        self, features_df: pd.DataFrame,
        patient_ids: Optional[list[str]] = None,
        with_lime: bool = True,
    ) -> dict[str, Any]:
        if with_lime:
            X_t = self.prep.transform(features_df[self.feature_names])
            self._fit_lime(np.asarray(X_t))

        reports: list[PatientReport] = []
        for i in range(len(features_df)):
            pid = patient_ids[i] if patient_ids else None
            reports.append(self.evaluate_patient(features_df.iloc[[i]], patient_id=pid))
            if (i + 1) % 10 == 0:
                logger.info("Evaluated %d / %d patients", i + 1, len(features_df))

        return {"reports": [r.to_dict() for r in reports], "summary": self._aggregate(reports)}

    @staticmethod
    def _aggregate(reports: list[PatientReport]) -> dict[str, Any]:
        def col(section, key):
            vals = [getattr(r, section).get(key) for r in reports if getattr(r, section)]
            vals = [v for v in vals if v is not None and v == v]
            return round(float(np.mean(vals)), 4) if vals else float("nan")

        return {
            "n": len(reports),
            "nl_faithfulness": {
                k: col("nl_faithfulness", k)
                for k in ["coverage", "fabrication", "direction_accuracy",
                          "direction_error", "direction_unclear", "rank_fidelity"]
            },
            "model_grounding": {
                k: col("model_grounding", k)
                for k in ["comprehensiveness", "sufficiency"]
            },
            "lime_agreement": {
                k: col("lime_agreement", k)
                for k in ["jaccard", "sign_agreement", "rank_tau"]
            },
        }
