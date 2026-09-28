"""Inject class imbalance into a clean binary dataset."""
import pandas as pd


def inject_imbalance(df, target_col="target", minority_share=0.1, random_state=None):
    """
    Return a copy of df where the minority class makes up roughly
    `minority_share` of the rows, by downsampling whichever class
    is too large relative to that target.

    Parameters
    ----------
    df : DataFrame with a binary target column.
    target_col : name of the target column.
    minority_share : desired fraction of rows belonging to the minority class.
    random_state : int, for reproducible sampling.

    Returns
    -------
    (flawed_df, info) where info is a dict describing what was done.
    """
    counts = df[target_col].value_counts()
    minority_class = counts.idxmin()
    majority_class = counts.idxmax()

    minority_rows = df[df[target_col] == minority_class]
    majority_rows = df[df[target_col] == majority_class]

    n_min = len(minority_rows)
    n_maj = len(majority_rows)
    natural_share = n_min / (n_min + n_maj)

    if minority_share <= natural_share:
        # Target is MORE imbalanced than natural -> shrink the minority class,
        # keep all majority rows.
        n_min_target = int(minority_share * n_maj / (1 - minority_share))
        n_min_target = min(n_min_target, n_min)
        minority_kept = minority_rows.sample(n=n_min_target, random_state=random_state)
        majority_kept = majority_rows
    else:
        # Target is LESS imbalanced than natural -> shrink the majority class,
        # keep all minority rows.
        n_maj_target = int(n_min * (1 - minority_share) / minority_share)
        n_maj_target = min(n_maj_target, n_maj)
        majority_kept = majority_rows.sample(n=n_maj_target, random_state=random_state)
        minority_kept = minority_rows

    flawed_df = pd.concat([minority_kept, majority_kept]).sample(frac=1, random_state=random_state)
    flawed_df = flawed_df.reset_index(drop=True)

    actual_share = len(minority_kept) / len(flawed_df)

    info = {
        "flaw": "imbalance",
        "target_minority_share": minority_share,
        "actual_minority_share": round(actual_share, 4),
        "minority_rows_kept": len(minority_kept),
        "majority_rows_kept": len(majority_kept),
    }
    return flawed_df, info