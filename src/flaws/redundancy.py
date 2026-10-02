"""Inject redundant (duplicate or near-duplicate) features into a clean dataset."""
import numpy as np
import pandas as pd


def inject_redundancy(df, target_col="target", n_redundant=3, noise_level=0.02, random_state=None):
    """
    Add extra columns that are near-copies of randomly chosen existing
    feature columns, with a small amount of noise so they aren't
    perfectly identical.

    Parameters
    ----------
    df : DataFrame with a binary target column.
    target_col : name of the target column, left untouched.
    n_redundant : how many new redundant columns to add.
    noise_level : how much noise to add to each copy, relative to the
        source column's standard deviation. 0 = exact duplicate.
    random_state : int, for reproducible column choice and noise.

    Returns
    -------
    (flawed_df, info) where info is a dict describing what was done.
    """
    rng = np.random.default_rng(random_state)
    flawed_df = df.copy()

    feature_cols = [c for c in df.columns if c != target_col]
    source_cols = rng.choice(feature_cols, size=n_redundant, replace=True)

    new_columns = {}
    correlations = []

    for i, source_col in enumerate(source_cols):
        noise = rng.normal(loc=0.0, scale=noise_level * df[source_col].std(), size=len(df))
        new_col_name = f"redundant_{i}_of_{source_col}"
        new_columns[new_col_name] = df[source_col] + noise
        correlations.append(round(float(np.corrcoef(df[source_col], new_columns[new_col_name])[0, 1]), 4))

    flawed_df = pd.concat([flawed_df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    info = {
        "flaw": "redundancy",
        "n_redundant": n_redundant,
        "source_columns": list(source_cols),
        "correlations_with_source": correlations,
    }
    return flawed_df, info