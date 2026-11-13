"""Train the final detectors on ALL the data and save them to one file (models/detectors.joblib).

The feature_noise detector is left out on purpose: against healthy data it scored an AUC of
only 0.58 (close to chance), so it mostly caused false alarms. The tool says so openly
instead of guessing.

Usage:
    python -m src.layer2.final_detectors
"""
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.layer2.train_detectors import FEATURE_COLS, FLAWS, LABEL_COL, load_meta

MODEL_FILE = Path("models/detectors.joblib")
RETIRED = ["feature_noise"]
SENSITIVITY = {"sensitive": 0.3, "balanced": 0.5, "strict": 0.7}   # when a detector "fires"


def main():
    df = load_meta()
    X = df[FEATURE_COLS]
    detectors = {}

    for flaw in FLAWS:
        if flaw in RETIRED:
            continue
        y_flaw = (df[LABEL_COL] == flaw).astype(int)
        model = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
        model.fit(X, y_flaw, sample_weight=compute_sample_weight("balanced", y_flaw))
        detectors[flaw] = model
        print(f"trained detector: {flaw}")

    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump({
        "detectors": detectors,
        "feature_cols": FEATURE_COLS,
        "retired": RETIRED,
        "sensitivity": SENSITIVITY,
        "n_training_rows": len(df),
    }, MODEL_FILE, compress=3)
    size_mb = MODEL_FILE.stat().st_size / 1e6
    print(f"\nSaved {len(detectors)} detectors (trained on {len(df)} rows) to {MODEL_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()