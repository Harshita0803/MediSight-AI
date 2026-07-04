import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


_PROMPT_A = """You are explaining a hospital readmission risk prediction to a clinician.
Use ONLY the risk factors listed below. Do not introduce any other clinical
factors. For each factor, state whether it increases or decreases the predicted
readmission risk, consistent with the sign given.

Predicted 30-day readmission probability: {risk:.0%}

Risk factors (feature, signed contribution; positive = increases risk):
{factors}

Write 3-5 sentences in plain clinical language."""


_PROMPT_B = """You are a clinician reviewing a patient's record to explain their
30-day hospital readmission risk. Below are the patient's actual data values.

You are NOT told which factors the risk model weighted. Using your own clinical
judgment, identify the factors that most drive THIS patient's readmission risk
up or down, and state the direction for each (increases or decreases risk).
Name the specific factors explicitly.

Predicted 30-day readmission probability: {risk:.0%}

Patient data:
{values}

Write 3-5 sentences in plain clinical language, naming the factors you rely on."""


def _fmt_factors(shap_top):
    return "\n".join(f"- {feat.replace('_', ' ')}: {val:+.3f}" for feat, val in shap_top)


def _fmt_values(feature_values: dict):
    lines = []
    for feat, val in feature_values.items():
        name = feat.replace("_", " ")
        if val is None:
            shown = "unknown"
        elif feat.startswith("has_") or feat == "is_first_admission":
            shown = "yes" if val and val >= 0.5 else "no"
        elif float(val).is_integer():
            shown = str(int(val))
        else:
            shown = f"{val:.2f}"
        lines.append(f"- {name}: {shown}")
    return "\n".join(lines)


def _make_llm(model, temperature):
    return ChatOpenAI(
        model=model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        api_key=os.environ["OPENAI_API_KEY"],
    )


def make_explainer(condition: str = "A", model: str | None = None, temperature: float = 0.0):
    condition = condition.upper()
    llm = _make_llm(model, temperature)

    if condition == "A":
        chain = ChatPromptTemplate.from_template(_PROMPT_A) | llm | StrOutputParser()
        def explain(shap_top, risk, feature_values=None):
            return chain.invoke({"risk": risk, "factors": _fmt_factors(shap_top)})
        return explain

    if condition == "B":
        chain = ChatPromptTemplate.from_template(_PROMPT_B) | llm | StrOutputParser()
        def explain(shap_top, risk, feature_values):
            return chain.invoke({"risk": risk, "values": _fmt_values(feature_values)})
        return explain

    raise ValueError(f"condition must be 'A' or 'B', got {condition!r}")


def make_openai_explainer(model: str | None = None, temperature: float = 0.0):
    return make_explainer("A", model=model, temperature=temperature)
