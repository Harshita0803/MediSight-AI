from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from evaluation.faithfulness import classify_feature, INCREASE, DECREASE, UNCLEAR
from evaluation.lexicon import SYNONYMS


def sample(results_path: Path, n: int, out_path: Path, seed: int = 42,
           n_distractors: int = 2) -> None:
    data = json.loads(results_path.read_text())
    reports = data["reports"]
    rng = random.Random(seed)
    rng.shuffle(reports)
    reports = reports[:n]

    feature_universe = list(SYNONYMS.keys())

    rows = []
    for r in reports:
        pid = r.get("patient_id") or ""
        text = r["explanation_text"]
        top_feats = [f for f, _ in r["shap_top"]]
        for feat, shap_val in r["shap_top"]:
            rows.append(_row(pid, feat, "increase" if shap_val > 0 else "decrease", text, "top_k"))
        distractors = [f for f in feature_universe if f not in top_feats]
        for feat in rng.sample(distractors, min(n_distractors, len(distractors))):
            rows.append(_row(pid, feat, "", text, "distractor"))

    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        f.write("# INSTRUCTIONS: for each row, read explanation_text and judge THIS feature only.\n")
        f.write("# human_mentioned = 1 (or yes) if the text refers to the feature, else 0 (or no).\n")
        f.write("# human_direction = increase/decrease/unclear (leave blank if human_mentioned=0).\n")
        f.write("# 'kind' is metadata only (top_k vs distractor) — ignore it while labeling.\n")
        w = csv.writer(f)
        w.writerow(list(rows[0].keys()))
        for row in rows:
            w.writerow(list(row.values()))
    n_top = sum(1 for r in rows if r["kind"] == "top_k")
    n_dis = sum(1 for r in rows if r["kind"] == "distractor")
    print(f"Wrote {len(rows)} rows ({n_top} top-k + {n_dis} distractor) "
          f"from {len(reports)} explanations -> {out_path}")
    print("Fill in human_mentioned and human_direction, then run: annotate score")


def _row(pid, feat, shap_sign, text, kind):
    return {
        "patient_id": pid, "feature": feat, "shap_sign": shap_sign,
        "kind": kind, "explanation_text": text,
        "human_mentioned": "", "human_direction": "",
    }


def _parse_human_mentioned(raw: str) -> int | None:
    v = raw.strip().lower()
    if v in {"1", "yes", "true"}:
        return 1
    if v in {"0", "no", "false"}:
        return 0
    return None


def score(sheet_path: Path) -> None:
    lines = [
        ln for ln in sheet_path.read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().lstrip('"').startswith("#")
    ]
    rows = list(csv.DictReader(lines))
    rows = [r for r in rows if (r.get("human_mentioned") or "").strip() != ""]
    if not rows:
        print("No labeled rows found. Fill in human_mentioned / human_direction first.")
        return

    human_ment, parser_ment = [], []
    human_dir, parser_dir = [], []
    misses = []

    for r in rows:
        hm = _parse_human_mentioned(r.get("human_mentioned", ""))
        if hm is None:
            continue
        pm, pd_ = classify_feature(r["explanation_text"], r["feature"])
        human_ment.append(hm)
        parser_ment.append(pm)
        if hm == 1 and pm == 0:
            misses.append((r["feature"], r["explanation_text"]))
        if hm == 1 and pm == 1:
            hd = (r.get("human_direction") or "").strip().lower() or UNCLEAR
            human_dir.append(hd)
            parser_dir.append(pd_)

    if not human_ment:
        print("No annotated rows found after parsing. Check human_mentioned values.")
        return

    if len(set(human_ment)) < 2:
        only = "mentioned=1" if (human_ment and human_ment[0] == 1) else "mentioned=0"
        print("=== Mention detection ===")
        print(f"  WARNING: every human label is {only}. Cohen's kappa is undefined")
        print("  with a single class. Re-sample WITH distractors so both classes appear.")
        p, rec, f1, _ = precision_recall_fscore_support(
            human_ment, parser_ment, average="binary", zero_division=0)
        print(f"  precision/recall/F1 (still valid): {p:.3f} / {rec:.3f} / {f1:.3f}")
    else:
        kappa_m = cohen_kappa_score(human_ment, parser_ment)
        p, rec, f1, _ = precision_recall_fscore_support(
            human_ment, parser_ment, average="binary", zero_division=0)
        print("=== Mention detection (parser vs human) ===")
        print(f"  n rows            : {len(rows)}")
        print(f"  Cohen's kappa     : {kappa_m:.3f}   (target >= 0.80)")
        print(f"  precision/recall/F1: {p:.3f} / {rec:.3f} / {f1:.3f}")

    if len(human_dir) >= 1 and len(set(human_dir)) >= 2:
        kappa_d = cohen_kappa_score(human_dir, parser_dir)
        raw = sum(a == b for a, b in zip(human_dir, parser_dir)) / len(human_dir)
        hbal = Counter(human_dir)
        print("\n=== Direction classification (on jointly-mentioned) ===")
        print(f"  n                 : {len(human_dir)}")
        print(f"  raw agreement     : {raw:.3f}")
        print(f"  Cohen's kappa     : {kappa_d:.3f}   (target >= 0.80)")
        print(f"  human class balance: " +
              ", ".join(f"{k}={v}" for k, v in hbal.most_common()))
        unclear_miss = sum(1 for h, pd in zip(human_dir, parser_dir)
                           if pd == UNCLEAR and h != UNCLEAR)
        if unclear_miss:
            print(f"  parser said 'unclear' on {unclear_miss} rows the human directed")
        dom = hbal.most_common(1)[0][1] / len(human_dir)
        if dom > 0.8 and raw > kappa_d + 0.3:
            print("  NOTE: one direction dominates >80%; report raw agreement alongside kappa.")
    elif human_dir:
        print("\n=== Direction classification ===")
        print(f"  Only one direction class present in {len(human_dir)} labels — kappa undefined.")
    else:
        print("\n(no jointly-mentioned rows to score direction)")

    if misses:
        by_feat = Counter(f for f, _ in misses)
        print(f"\n=== Parser misses (human=mentioned, parser=missed): {len(misses)} ===")
        for feat, cnt in by_feat.most_common(8):
            print(f"  {cnt:3d}x  {feat}")
        print("\nExample missed sentences:")
        for feat, text in misses[:5]:
            snippet = text[:160].replace("\n", " ")
            print(f"  [{feat}] ...{snippet}...")

    print("\nInterpretation: mention-detection kappa >= 0.80 = defensible metrics.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--results", type=Path, default=Path("reports/faithfulness_results.json"))
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--out", type=Path, default=Path("reports/annotation_sheet.csv"))
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--distractors", type=int, default=2)

    c = sub.add_parser("score")
    c.add_argument("--sheet", type=Path, default=Path("reports/annotation_sheet.csv"))

    args = ap.parse_args()
    if args.cmd == "sample":
        sample(args.results, args.n, args.out, args.seed, args.distractors)
    else:
        score(args.sheet)


if __name__ == "__main__":
    main()
