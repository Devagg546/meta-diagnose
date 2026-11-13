"""Layer 1 (v2): run every condition on several random slices of every dataset.

Why slices? Every dataset has its own idea of "normal". Showing the Doctor many different
slices (random rows and columns) of every dataset teaches it a wider range of "normal".
The `dataset` column always holds the BASE dataset name, so the Doctor is still tested on
whole datasets it has never seen.

Usage:
    python -m src.layer1.run_experiments           # the full run
    python -m src.layer1.run_experiments --quick   # tiny test run (about a minute)
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.flaws.imbalance import inject_imbalance
from src.layer1.conditions import build_condition_list, apply_condition
from src.layer1.meta_features import (
    dataset_meta_features,
    model_meta_features,
    error_meta_features,
    probe_meta_features,
)

# Divide-by-zero warnings from constant columns are harmless here (NaNs are ignored in the averages).
warnings.filterwarnings("ignore", category=RuntimeWarning)

RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")

MAX_ROWS = 3000         # no slice is larger than this, to keep runs fast
N_RANDOM_SLICES = 2     # random slices per dataset, on top of one "full" slice
RANDOM_STATE = 0
HEALTHY_MIN_SHARE = 0.2  # a healthy dataset has at least this much minority class


def stratified_rows(df, n_rows, seed):
    """Pick n_rows rows at random while keeping the class balance."""
    if n_rows >= len(df):
        return df.reset_index(drop=True)
    sub, _ = train_test_split(df, train_size=n_rows, stratify=df["target"], random_state=seed)
    return sub.reset_index(drop=True)


def make_slices(clean_df, rng, n_random):
    """One 'full' slice plus n_random slices.

    Each random slice keeps a random share of the rows and columns, and is also re-balanced to a
    random class balance inside the healthy range (20% to 50% minority). That teaches the Doctor
    that many different balances are normal, and that "imbalance" starts below that range.
    """
    feature_cols = [c for c in clean_df.columns if c != "target"]
    n_feat = len(feature_cols)

    slices = [("full", stratified_rows(clean_df, MAX_ROWS, int(rng.integers(2**31 - 1))))]

    for i in range(1, n_random + 1):
        n_rows = int(len(clean_df) * rng.uniform(0.5, 1.0))
        n_rows = min(MAX_ROWS, max(n_rows, min(len(clean_df), 150)))

        low = min(n_feat, max(3, int(np.ceil(n_feat * 0.4))))
        n_cols = int(rng.integers(low, n_feat + 1))
        chosen = set(rng.choice(feature_cols, size=n_cols, replace=False))
        cols = [c for c in feature_cols if c in chosen]

        sub = stratified_rows(clean_df, n_rows, int(rng.integers(2**31 - 1)))
        sub = sub[cols + ["target"]]

        healthy_share = rng.uniform(HEALTHY_MIN_SHARE, 0.5)
        rebalanced, _ = inject_imbalance(sub, minority_share=healthy_share,
                                         random_state=int(rng.integers(2**31 - 1)))
        slices.append((f"slice{i}", rebalanced if len(rebalanced) >= 100 else sub))
    return slices


def run_one(dataset_name, slice_name, clean_df, condition, random_state):
    """Break one slice in one way, train one model, and return one row of symptoms."""
    flawed_df, _ = apply_condition(clean_df, condition, random_state=random_state)

    feature_cols = [c for c in flawed_df.columns if c != "target"]
    X = flawed_df[feature_cols]
    y = flawed_df["target"]
    X = X.fillna(X.mean(numeric_only=True)).fillna(0.0)   # models cannot read NaN

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)

    row = {
        "dataset": dataset_name,          # BASE dataset: this is what the Doctor is tested on
        "slice": slice_name,
        "condition_id": condition["condition_id"],
        "true_flaw": condition["flaw"],
        "model": "random_forest",
    }
    row.update(dataset_meta_features(flawed_df))
    row.update(probe_meta_features(flawed_df))
    row.update(model_meta_features(y_train, y_train_pred, y_test, y_test_pred))
    row.update(error_meta_features(y_test, y_test_pred, y_test_proba))
    return row


def main(quick=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conditions = build_condition_list()
    names = sorted(p.stem for p in RAW_DIR.glob("*.csv"))
    n_random = N_RANDOM_SLICES
    out_file = RESULTS_DIR / "meta_dataset_v2.csv"

    if quick:
        names = names[:2]
        conditions = conditions[::4]      # clean + one setting of every flaw
        n_random = 1
        out_file = RESULTS_DIR / "meta_dataset_v2_quick.csv"

    total = len(names) * (1 + n_random) * len(conditions)
    print(f"{len(names)} datasets x {1 + n_random} slices x {len(conditions)} conditions = {total} rows")
    print("datasets:", ", ".join(names), "\n")

    results, failures = [], 0
    start = time.time()

    for ds_index, name in enumerate(names):
        clean_df = pd.read_csv(RAW_DIR / f"{name}.csv")
        rng = np.random.default_rng([RANDOM_STATE, ds_index])

        for slice_name, slice_df in make_slices(clean_df, rng, n_random):
            for condition in conditions:
                try:
                    results.append(run_one(name, slice_name, slice_df, condition, RANDOM_STATE))
                except Exception as e:
                    failures += 1
                    print(f"FAILED: {name} | {slice_name} | {condition['condition_id']} -> {e}")

        pd.DataFrame(results).to_csv(out_file, index=False)   # save as we go
        print(f"Finished {name}: {len(results)} rows so far ({time.time() - start:.0f}s)")

    out = pd.DataFrame(results)
    print(f"\nDone. Saved {len(out)} rows to {out_file} (failures: {failures})")

    nan_counts = out.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        print("\nColumns containing NaN (tell me if you see this):")
        print(nan_counts.to_string())
    print("\nRows per flaw:")
    print(out["true_flaw"].value_counts().to_string())


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)