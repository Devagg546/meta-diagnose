"""The final exam: run the frozen detectors on real datasets that were never part of training.

Ground truth that we can check objectively from the raw data:
  - imbalance:      the smaller class is under 17% of the rows (the mildest imbalance we trained on)
  - missing values: at least 5% of the cells are empty (the mildest missing rate we trained on)
Alarms from the other detectors cannot be judged right or wrong on natural data, so they are listed only.

Usage:
    python -m src.layer2.natural_exam
"""
import warnings
from pathlib import Path

import joblib
import pandas as pd

from src.diagnose import load_table, prepare
from src.layer1.conditions import build_condition_list
from src.layer1.run_experiments import MAX_ROWS, stratified_rows, run_one

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 220)

NATURAL_DIR = Path("data/natural")
MODEL_FILE = Path("models/detectors.joblib")
RESULTS_DIR = Path("results")

IMBALANCED_BELOW = 0.17   # the mildest imbalance the detectors were trained on
HEALTHY_FROM = 0.20       # at least this much minority class counts as a healthy balance
MISSING_FROM = 0.05       # the smallest missing rate the detectors were trained on
MISSING_NONE_BELOW = 0.01


def main():
    bundle = joblib.load(MODEL_FILE)
    rows = []

    for path in sorted(NATURAL_DIR.glob("*.csv")):
        try:
            data, _, _ = prepare(load_table(path), "target")
            n_total = len(data)
            data = stratified_rows(data, MAX_ROWS, 0)
            measured = run_one(path.stem, "full", data, build_condition_list()[0], 0)   # same measurements as training
        except (Exception, SystemExit) as e:
            print(f"SKIPPED {path.stem}: {e}")
            continue

        X = pd.DataFrame([measured])[bundle["feature_cols"]]
        probs = {f: float(m.predict_proba(X)[0, 1]) for f, m in bundle["detectors"].items()}
        rows.append({"dataset": path.stem, "rows": n_total, "features": data.shape[1] - 1,
                     "minority": measured["minority_class_share"], "missing": measured["missing_rate"], **probs})

    if not rows:
        raise SystemExit("No datasets found. Run:  python -m src.data_loader_natural")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "natural_exam.csv", index=False)
    detectors = list(bundle["detectors"])

    print("=== Every dataset: what we measured, and each detector's confidence ===")
    show = res.copy()
    show["minority"] = (show["minority"] * 100).round(1).astype(str) + "%"
    show["missing"] = (show["missing"] * 100).round(1).astype(str) + "%"
    print(show.round(2).to_string(index=False))

    print("\n=== Imbalance detector (truth: smaller class under 17%) ===")
    is_imb = res["minority"] < IMBALANCED_BELOW
    is_bal = res["minority"] >= HEALTHY_FROM
    for t in [0.3, 0.5, 0.7]:
        print(f"  fires at {t}: {int((res.loc[is_imb, 'imbalance'] >= t).sum())} of {int(is_imb.sum())} imbalanced datasets;"
              f"  false alarms: {int((res.loc[is_bal, 'imbalance'] >= t).sum())} of {int(is_bal.sum())} balanced datasets")

    print("\n=== Missing-values detector (truth: 5% or more of the cells empty) ===")
    has_miss = res["missing"] >= MISSING_FROM
    no_miss = res["missing"] < MISSING_NONE_BELOW
    for t in [0.3, 0.5, 0.7]:
        print(f"  fires at {t}: {int((res.loc[has_miss, 'missing_values'] >= t).sum())} of {int(has_miss.sum())} datasets with missing values;"
              f"  false alarms: {int((res.loc[no_miss, 'missing_values'] >= t).sum())} of {int(no_miss.sum())} datasets without")

    print("\n=== Other detectors at 0.5 (no ground truth on natural data: these alarms may well be real) ===")
    for flaw in detectors:
        if flaw in ("imbalance", "missing_values"):
            continue
        hit = res.loc[res[flaw] >= 0.5, "dataset"].tolist()
        print(f"  {flaw:12s} {len(hit):2d} of {len(res)}: {', '.join(hit) if hit else '-'}")


if __name__ == "__main__":
    main()