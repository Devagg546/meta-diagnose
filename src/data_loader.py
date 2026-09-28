"""Download the base datasets and save them as clean CSVs in data/raw/."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer

RAW_DIR = Path("data/raw")

# short name we use in the project -> name on OpenML
OPENML_DATASETS = {
    "spambase": "spambase",
    "banknote": "banknote-authentication",
    "phoneme": "phoneme",
    "ionosphere": "ionosphere",
    "diabetes": "diabetes",
    "magic": "MagicTelescope",
}


def to_binary_target(y):
    """Turn any label column (strings or numbers) into 0/1 integers."""
    codes, _ = pd.factorize(y, sort=True)
    return pd.Series(codes, index=y.index, name="target")


def save(name, X, y):
    """Attach the target, save to CSV, and print a quick summary."""
    assert X.select_dtypes(exclude="number").empty, f"{name} has non-numeric columns"
    assert y.nunique() == 2, f"{name} target has {y.nunique()} classes, expected 2"

    df = X.copy()
    df["target"] = y.values
    df.to_csv(RAW_DIR / f"{name}.csv", index=False)

    minority_share = df["target"].value_counts(normalize=True).min()
    print(f"{name:14s} rows={len(df):5d}  features={X.shape[1]:3d}  "
          f"minority share={minority_share:.2f}  missing cells={df.isna().sum().sum()}")


def load_shuttle_binary():
    """Shuttle is naturally 7-class. Fold it into binary: class 1 (normal) vs rest (anomaly)."""
    d = fetch_openml(name="shuttle", version=1, as_frame=True)
    is_anomaly = (d.target.astype(int) != 1).astype(int)
    return d.data, pd.Series(is_anomaly.values, index=d.data.index, name="target")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    bc = load_breast_cancer(as_frame=True)   # ships with scikit-learn, no download
    save("breast_cancer", bc.data, to_binary_target(bc.target))

    for name, openml_name in OPENML_DATASETS.items():
        d = fetch_openml(name=openml_name, version=1, as_frame=True)
        save(name, d.data, to_binary_target(d.target))

    X, y = load_shuttle_binary()
    save("shuttle", X, y)


if __name__ == "__main__":
    main()