"""Diagnose a dataset: which of the known flaws does it show?

Usage:
    python -m src.diagnose path/to/data.csv
    python -m src.diagnose path/to/data.csv --target label --sensitivity strict

Works on CSV, Excel, JSON and Parquet files. The dataset needs one target column with
exactly two values. Non-numeric feature columns are ignored (v1 handles numeric data only).
"""
import argparse
import warnings
from pathlib import Path

import joblib
import pandas as pd

from src.layer1.conditions import build_condition_list
from src.layer1.run_experiments import MAX_ROWS, stratified_rows, run_one

warnings.filterwarnings("ignore", category=RuntimeWarning)

MODEL_FILE = Path("models/detectors.joblib")

HINTS = {
    "imbalance": "one class is rare. Common fixes: class weights, resampling (e.g. SMOTE), or judging the model by recall/F1 instead of accuracy.",
    "missing_values": "many cells are empty. Common fixes: imputation, or dropping columns that are mostly empty.",
    "leakage": "a feature seems to reveal the answer. Check where each column comes from, and remove any that would not exist at prediction time.",
    "label_noise": "some labels seem wrong. Review the rows the model is most sure are mislabelled.",
    "outliers": "extreme values are present. Check for entry errors; consider clipping or robust scaling.",
    "redundancy": "some columns are near-copies of others. Consider dropping duplicates.",
}


def load_table(path):
    suffix = Path(path).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def prepare(df, target):
    """Turn an arbitrary table into the format the pipeline expects: numeric features + a 0/1 'target'."""
    if target is None:
        target = "target" if "target" in df.columns else df.columns[-1]
    if target not in df.columns:
        raise SystemExit(f"Column '{target}' not found. Columns are: {list(df.columns)}")

    n_before = len(df)
    df = df.dropna(subset=[target])
    if len(df) < n_before:
        print(f"Note: dropped {n_before - len(df)} rows with an empty target.")

    y = df[target]
    if y.nunique() != 2:
        raise SystemExit(f"Target column '{target}' has {y.nunique()} distinct values; this tool needs exactly 2.")

    X = df.drop(columns=[target])
    bool_cols = [c for c in X.columns if X[c].dtype == bool]
    X = X.astype({c: int for c in bool_cols})
    numeric = X.select_dtypes(include="number")
    ignored = [c for c in X.columns if c not in numeric.columns]
    numeric = numeric.loc[:, numeric.nunique() > 1]        # columns that never change carry no information
    if numeric.shape[1] < 3:
        raise SystemExit("Need at least 3 numeric feature columns (that are not constant).")

    out = numeric.reset_index(drop=True).copy()
    out["target"] = pd.factorize(y, sort=True)[0]
    return out, ignored, target


def main():
    parser = argparse.ArgumentParser(description="Diagnose a tabular dataset for common data problems.")
    parser.add_argument("path", help="CSV / Excel / JSON / Parquet file")
    parser.add_argument("--target", help="name of the target column (default: 'target', else the last column)")
    parser.add_argument("--sensitivity", choices=["sensitive", "balanced", "strict"], default="balanced",
                        help="sensitive = catch more problems, more false alarms; strict = fewer false alarms")
    args = parser.parse_args()

    if not MODEL_FILE.exists():
        raise SystemExit("No trained detectors found. Run:  python -m src.layer2.final_detectors")
    bundle = joblib.load(MODEL_FILE)
    threshold = bundle["sensitivity"][args.sensitivity]

    raw = load_table(args.path)
    data, ignored, target = prepare(raw, args.target)
    n_rows_total = len(data)
    data = stratified_rows(data, MAX_ROWS, 0)

    try:
        row = run_one("uploaded", "full", data, build_condition_list()[0], 0)    # the same measurements as in training
    except ValueError as e:
        raise SystemExit(f"Could not analyse this dataset (is one class extremely rare?): {e}")

    X = pd.DataFrame([row])[bundle["feature_cols"]]
    probs = {flaw: float(model.predict_proba(X)[0, 1]) for flaw, model in bundle["detectors"].items()}
    fired = sorted([f for f, p in probs.items() if p >= threshold], key=lambda f: -probs[f])

    print(f"\nDataset: {Path(args.path).name}")
    note = f" (analysed a random sample of {len(data)})" if len(data) < n_rows_total else ""
    print(f"  {n_rows_total} rows{note}, {data.shape[1] - 1} numeric features, target = '{target}'")
    if ignored:
        print(f"  ignored non-numeric columns: {ignored}")
    print("\nKey measurements:")
    print(f"  smaller class share      {row['minority_class_share']:.1%}")
    print(f"  missing cells            {row['missing_rate']:.1%}")
    print(f"  strongest feature->target link   {row['max_feature_target_correlation']:.2f}")
    print(f"  strongest link between two features  {row['max_pairwise_correlation']:.2f}")
    print(f"  how much a quick model believes the labels  {row['mean_given_label_prob']:.2f}")

    print(f"\nDetectors ({args.sensitivity}: a detector fires at {threshold:.2f} or higher)")
    for flaw, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        mark = "  <-- FIRED" if flaw in fired else ""
        print(f"  {flaw:15s} {p:4.2f}  {'#' * int(round(p * 20)):20s}{mark}")

    print()
    if fired:
        print("Likely problems: " + ", ".join(fired))
        for flaw in fired:
            print(f"  - {flaw}: {HINTS[flaw]}")
    else:
        print("No clear problem found.")
    print("\nLimits: weak leakage, light label noise and any feature noise cannot be diagnosed reliably "
          "from a single dataset, so 'no clear problem' is not a guarantee.")


if __name__ == "__main__":
    main()