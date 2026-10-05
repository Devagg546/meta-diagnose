"""Layer 2: train the Doctor Model and compare it with baselines,
using leave-one-dataset-out cross-validation."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

RESULTS_DIR = Path("results")

FEATURE_COLS = [
    "minority_class_share", "missing_rate", "mean_feature_correlation",
    "max_feature_target_correlation", "outlier_score", "n_rows", "n_features",
    "accuracy", "macro_f1", "minority_recall", "train_test_gap",
    "error_rate_class_0", "error_rate_class_1", "mean_confidence_on_errors",
]
LABEL_COL = "true_flaw"
GROUP_COL = "dataset"


def rule_based_diagnosis(X):
    """Hand-written baseline: simple thresholds, checked in order."""
    diagnoses = []
    for _, row in X.iterrows():
        if row["minority_class_share"] < 0.17:
            diagnoses.append("imbalance")
        elif row["missing_rate"] > 0.01:
            diagnoses.append("missing_values")
        elif row["max_feature_target_correlation"] > 0.9:
            diagnoses.append("leakage")
        elif row["outlier_score"] > 0.04:
            diagnoses.append("outliers")
        else:
            diagnoses.append("clean")
    return diagnoses


def make_models():
    """Factories, so every fold gets a fresh untrained model."""
    return {
        "majority_class": lambda: DummyClassifier(strategy="most_frequent"),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, random_state=0),
        "xgboost": lambda: XGBClassifier(eval_metric="mlogloss", random_state=0),
    }


def main():
    df = pd.read_csv(RESULTS_DIR / "meta_dataset.csv")
    X = df[FEATURE_COLS]
    groups = df[GROUP_COL]

    encoder = LabelEncoder()
    y = encoder.fit_transform(df[LABEL_COL])

    models = make_models()
    diagnoser_names = ["rules"] + list(models.keys())
    all_preds = {name: np.zeros(len(df), dtype=int) for name in diagnoser_names}
    fold_rows = []

    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        held_out = groups.iloc[test_idx].iloc[0]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        preds = {"rules": encoder.transform(rule_based_diagnosis(X_test))}
        for name, make_model in models.items():
            model = make_model()
            model.fit(X_train, y_train)
            preds[name] = model.predict(X_test)

        for name, y_pred in preds.items():
            all_preds[name][test_idx] = y_pred
            fold_rows.append({
                "held_out_dataset": held_out,
                "diagnoser": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            })

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(RESULTS_DIR / "doctor_cv_by_dataset.csv", index=False)

    print("=== Mean +/- std across the 8 held-out datasets ===")
    summary = fold_df.groupby("diagnoser")[["accuracy", "macro_f1"]].agg(["mean", "std"])
    print(summary.round(3))

    print("\n=== Accuracy on each held-out dataset ===")
    pivot = fold_df.pivot(index="held_out_dataset", columns="diagnoser", values="accuracy")
    print(pivot.round(3))

    print("\n=== Random Forest confusion matrix (all folds pooled) ===")
    cm = confusion_matrix(y, all_preds["random_forest"])
    cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
    print(cm_df.to_string())


if __name__ == "__main__":
    main()