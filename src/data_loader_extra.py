"""Download extra base datasets (v2). A dataset that fails any check is skipped, with the reason printed."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

RAW_DIR = Path("data/raw")
MIN_MINORITY_SHARE = 0.2   # "clean" data should not already be heavily imbalanced

# short name we use in the project -> name on OpenML
EXTRA_DATASETS = {
    "blood_transfusion": "blood-transfusion-service-center",
    "heart_statlog": "heart-statlog",
    "sonar": "sonar",
    "qsar_biodeg": "qsar-biodeg",
    "eeg_eye_state": "eeg-eye-state",
    "ringnorm": "ringnorm",
    "twonorm": "twonorm",
    "credit_default": "default-of-credit-card-clients",
    "steel_plates": "steel-plates-fault",
}


def prepare(X, y):
    """Return (X, target) if the dataset passes every check, otherwise raise ValueError with the reason."""
    if not X.select_dtypes(exclude="number").empty:
        raise ValueError("has non-numeric columns")
    if y.nunique() != 2:
        raise ValueError(f"target has {y.nunique()} classes, expected 2")
    if X.isna().sum().sum() > 0:
        raise ValueError("has missing values")

    X = X.loc[:, X.nunique() > 1]          # drop columns that never change
    if X.shape[1] < 3:
        raise ValueError("fewer than 3 usable features")

    target = pd.Series(pd.factorize(y, sort=True)[0], index=X.index, name="target")
    share = target.value_counts(normalize=True).min()
    if share < MIN_MINORITY_SHARE:
        raise ValueError(f"minority share {share:.2f} is below {MIN_MINORITY_SHARE}")
    return X, target


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []

    for name, openml_name in EXTRA_DATASETS.items():
        try:
            d = fetch_openml(name=openml_name, as_frame=True)
            X, target = prepare(d.data, d.target)
            df = X.copy()
            df["target"] = target.values
            df.to_csv(RAW_DIR / f"{name}.csv", index=False)
            share = target.value_counts(normalize=True).min()
            print(f"SAVED   {name:20s} rows={len(df):6d}  features={X.shape[1]:3d}  minority share={share:.2f}")
            saved.append(name)
        except Exception as e:
            print(f"SKIPPED {name:20s} -> {e}")
            skipped.append(name)

    print(f"\n{len(saved)} saved, {len(skipped)} skipped.")


if __name__ == "__main__":
    main()