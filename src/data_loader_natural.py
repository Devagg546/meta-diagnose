"""Download real-world datasets that are NATURALLY flawed, for the final exam of the detectors.

Nothing here was used for training. Unlike the training loader, naturally imbalanced datasets
and datasets with real missing values are welcome. Anything that fails a check is skipped,
with the reason printed. The files go to data/natural/ (not data/raw/, which is the training set).

Usage:
    python -m src.data_loader_natural
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from src.diagnose import prepare

NATURAL_DIR = Path("data/natural")

# short name we use -> name on OpenML
NATURAL_DATASETS = {
    # software-defect data: usually heavily imbalanced
    "cm1": "cm1",
    "kc1": "kc1",
    "kc2": "kc2",
    "pc1": "pc1",
    "pc3": "pc3",
    "pc4": "pc4",
    "jm1": "jm1",
    # other imbalanced data
    "wilt": "wilt",
    "climate_crashes": "climate-model-simulation-crashes",
    "mammography": "mammography",
    "ozone": "ozone-level-8hr",
    "musk": "musk",
    # real missing values (numeric columns only are used)
    "colic": "colic",
    "hepatitis": "hepatitis",
    "sick": "sick",
    "hypothyroid": "hypothyroid",
    "ilpd": "ilpd",
    # reasonably balanced, never used in training
    "hill_valley": "hill-valley",
    "madelon": "madelon",
    "haberman": "haberman",
    "wholesale": "wholesale-customers",
    "planning_relax": "planning-relax",
}


def main():
    NATURAL_DIR.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []

    for name, openml_name in NATURAL_DATASETS.items():
        try:
            d = fetch_openml(name=openml_name, as_frame=True)
            df = d.data.copy()
            df["target"] = d.target
            data, ignored, _ = prepare(df, "target")          # numeric columns + a 0/1 target; NaN cells are kept
            data.to_csv(NATURAL_DIR / f"{name}.csv", index=False)
            share = data["target"].value_counts(normalize=True).min()
            missing = data.drop(columns="target").isna().mean().mean()
            print(f"SAVED   {name:16s} rows={len(data):6d}  features={data.shape[1] - 1:4d}  "
                  f"minority={share:5.1%}  missing={missing:5.1%}  (ignored {len(ignored)} non-numeric columns)")
            saved.append(name)
        except (Exception, SystemExit) as e:                    # prepare() exits with a message when a dataset does not fit
            print(f"SKIPPED {name:16s} -> {e}")
            skipped.append(name)

    print(f"\n{len(saved)} saved, {len(skipped)} skipped.")


if __name__ == "__main__":
    main()