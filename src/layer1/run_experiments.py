"""Run the full Layer 1 experiment grid: 8 datasets x 29 conditions x 3 models."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.layer1.conditions import build_condition_list, apply_condition
from src.layer1.meta_features import (
    dataset_meta_features,
    model_meta_features,
    error_meta_features,
)

RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")

DATASET_NAMES = [
    "breast_cancer", "spambase", "banknote", "phoneme",
    "ionosphere", "diabetes", "magic", "shuttle",
]

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=5000),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=0),
    "xgboost": XGBClassifier(eval_metric="logloss", random_state=0),
}


def run_one(dataset_name, clean_df, condition, model_name, model, random_state):
    """Run a single (dataset, condition, model) experiment. Returns one result row."""
    flawed_df, flaw_info = apply_condition(clean_df, condition, random_state=random_state)

    feature_cols = [c for c in flawed_df.columns if c != "target"]
    X = flawed_df[feature_cols]
    y = flawed_df["target"]

    # Some flaws (missing values) leave NaNs that most models can't train on directly.
    X = X.fillna(X.mean(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)

    row = {
        "dataset": dataset_name,
        "condition_id": condition["condition_id"],
        "true_flaw": condition["flaw"],
        "model": model_name,
    }
    row.update(dataset_meta_features(flawed_df))
    row.update(model_meta_features(y_train, y_train_pred, y_test, y_test_pred))
    row.update(error_meta_features(y_test, y_test_pred, y_test_proba))
    return row


def main(random_state=0):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conditions = build_condition_list()
    results = []

    for dataset_name in DATASET_NAMES:
        clean_df = pd.read_csv(RAW_DIR / f"{dataset_name}.csv")

        for condition in conditions:
            for model_name, model in MODELS.items():
                try:
                    row = run_one(dataset_name, clean_df, condition, model_name, model, random_state)
                    results.append(row)
                except Exception as e:
                    print(f"FAILED: {dataset_name} | {condition['condition_id']} | {model_name} -> {e}")

        print(f"Finished {dataset_name}: {len(results)} rows so far")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "meta_dataset.csv", index=False)
    print(f"\nDone. Saved {len(results_df)} rows to {RESULTS_DIR / 'meta_dataset.csv'}")


if __name__ == "__main__":
    main()