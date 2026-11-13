"""Layer 2, Try 3: one yes/no detector per flaw. "Clean" simply means that no detector fired.

Why? A single multi-class Doctor must always name a flaw, so it has no honest way to say
"I see nothing wrong". Here every detector answers its own question ("is there leakage?"),
and a dataset can trigger several detectors, or none at all.
Every detector is tested on datasets it has never seen (leave-one-dataset-out).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight

from src.layer2.train_doctor import FEATURE_COLS, LABEL_COL, GROUP_COL

pd.set_option("display.width", 200)

RESULTS_DIR = Path("results")
META_FILE = RESULTS_DIR / "meta_dataset_v2.csv"
EXTRA_FILE = RESULTS_DIR / "clean_extra_v2.csv"   # extra healthy slices (made by run_clean_extra)

FLAWS = ["imbalance", "missing_values", "leakage", "feature_noise",
         "label_noise", "outliers", "redundancy"]
THRESHOLD = 0.5      # a detector "fires" when its probability is at least this


def load_meta():
    """The main results, plus the extra healthy slices if they exist."""
    df = pd.read_csv(META_FILE)
    if EXTRA_FILE.exists():
        df = pd.concat([df, pd.read_csv(EXTRA_FILE)], ignore_index=True)
    return df


def headline(df, truth, guess, mask):
    """Accuracy and macro-F1, averaged over datasets, for the rows selected by mask."""
    scores = []
    for ds in df.loc[mask, GROUP_COL].unique():
        m = mask & (df[GROUP_COL] == ds)
        scores.append((accuracy_score(truth[m], guess[m]),
                       f1_score(truth[m], guess[m], average="macro", zero_division=0)))
    scores = np.array(scores)
    return (f"accuracy {scores[:, 0].mean():.3f} +/- {scores[:, 0].std(ddof=1):.3f}   "
            f"macro-F1 {scores[:, 1].mean():.3f} +/- {scores[:, 1].std(ddof=1):.3f}")


def detector_probabilities(df):
    """For every row, the probability of each flaw, from detectors that never saw that row's dataset."""
    X = df[FEATURE_COLS]
    groups = df[GROUP_COL]
    probs = np.zeros((len(df), len(FLAWS)))

    for train_idx, test_idx in LeaveOneGroupOut().split(X, groups=groups):
        for j, flaw in enumerate(FLAWS):
            y_flaw = (df[LABEL_COL] == flaw).to_numpy().astype(int)
            model = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
            model.fit(X.iloc[train_idx], y_flaw[train_idx],
                      sample_weight=compute_sample_weight("balanced", y_flaw[train_idx]))
            probs[test_idx, j] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return pd.DataFrame(probs, index=df.index, columns=FLAWS)


def diagnose(probs, threshold):
    """Detectors at or above the threshold 'fire'. Nothing fired -> 'clean'. Otherwise the strongest one wins."""
    fired = probs >= threshold
    top = probs.idxmax(axis=1)
    guess = pd.Series(np.where(fired.any(axis=1), top, "clean"), index=probs.index)
    return guess, fired


def own_detector_fired(fired, truth):
    """For every flawed row, did the detector for its TRUE flaw fire?"""
    flawed = truth != "clean"
    cols = truth[flawed].map({f: i for i, f in enumerate(FLAWS)}).to_numpy()
    return fired.to_numpy()[np.where(flawed)[0], cols]


def severity_levels(df):
    """Severity level 1..4 for every flawed row (1 = mildest), read from the end of condition_id."""
    setting = pd.to_numeric(df["condition_id"].str.split("_").str[-1], errors="coerce")
    level = pd.Series(np.nan, index=df.index)
    for flaw in FLAWS:
        m = df[LABEL_COL] == flaw
        values = sorted(setting[m].unique(), reverse=(flaw == "imbalance"))
        level[m] = setting[m].map({v: i + 1 for i, v in enumerate(values)})
    return level


def main():
    df = load_meta()
    truth = df[LABEL_COL]
    is_clean = truth == "clean"
    is_extra = df["slice"].astype(str).str.startswith("cleanx") if "slice" in df.columns else pd.Series(False, index=df.index)
    original = ~is_extra

    probs = detector_probabilities(df)
    guess, fired = diagnose(probs, THRESHOLD)

    keep = [c for c in [GROUP_COL, "slice", "condition_id", LABEL_COL] if c in df.columns]
    pd.concat([df[keep], probs.add_prefix("p_")], axis=1).to_csv(
        RESULTS_DIR / "detector_probabilities.csv", index=False)

    # ---- 1. headline, scored like the old Doctor (one answer per row) ----
    print("=== 1. Headline: strongest detector wins, 'clean' if none fired (mean +/- std over datasets) ===")
    print("original rows only (comparable with last time):", headline(df, truth, guess, original))
    print("all rows, incl. extra healthy slices:          ", headline(df, truth, guess, original | is_extra))
    print("(the single multi-class Random Forest Doctor scored 0.757 accuracy / 0.666 macro-F1)")

    # ---- 2. can it finally say 'nothing found'? ----
    print("\n=== 2. Healthy data ===")
    for label, m in [("original clean rows", is_clean & original), ("extra healthy slices", is_clean & is_extra)]:
        if m.sum():
            print(f"{label}: no detector fired on {(guess[m] == 'clean').sum()} of {m.sum()}")
    print("Detectors that fired on healthy rows (all):", fired[is_clean].sum().to_dict())
    print(f"Rows where more than one detector fired: {(fired.sum(axis=1) > 1).mean():.0%}")

    # ---- 3. each detector on its own ----
    rows = []
    for flaw in FLAWS:
        y_flaw = (truth == flaw).astype(int)
        rows.append({
            "detector": flaw,
            "AUC": roc_auc_score(y_flaw, probs[flaw]),
            "recall": fired.loc[truth == flaw, flaw].mean(),
            "precision": (truth[fired[flaw]] == flaw).mean() if fired[flaw].any() else np.nan,
            "false_alarm_rate": fired.loc[truth != flaw, flaw].mean(),
            "false_alarm_on_clean": fired.loc[is_clean, flaw].mean(),
        })
    print("\n=== 3. Each detector on its own (threshold %.1f) ===" % THRESHOLD)
    print(pd.DataFrame(rows).set_index("detector").round(2).to_string())
    print("AUC: 0.5 = no skill, 1.0 = perfect. recall = flawed rows caught. "
          "precision = alarms that were right.")

    # ---- 4. by severity ----
    level = severity_levels(df)
    table = {}
    for flaw in FLAWS:
        m = truth == flaw
        table[flaw] = fired.loc[m, flaw].groupby(level[m]).mean()
    print("\n=== 4. Share of flawed rows caught, by severity (1 = mildest, 4 = most severe) ===")
    print(pd.DataFrame(table).T.rename(columns=int).round(2).to_string())

    # ---- 5. what the threshold trades off (for understanding, not for tuning on this data) ----
    rows = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        g, f = diagnose(probs, t)
        rows.append({
            "threshold": t,
            "clean_called_clean": (g[is_clean] == "clean").mean(),
            "flawed_rows_caught": own_detector_fired(f, truth).mean(),
            "accuracy_one_answer": accuracy_score(truth, g),
        })
    print("\n=== 5. What the 'fire' threshold trades off (all rows; healthy rows are a bigger share now) ===")
    print(pd.DataFrame(rows).set_index("threshold").round(2).to_string())


if __name__ == "__main__":
    main()