"""Inject missing values into a clean dataset."""
import numpy as np
import pandas as pd


def inject_missing(df, target_col="target", missing_rate=0.1, random_state=None):
    """
    Return a copy of df where roughly `missing_rate` fraction of cells
    in the feature columns (never the target) are replaced with NaN.

    Parameters
    ----------
    df : DataFrame with a binary target column.
    target_col : name of the target column, left untouched.
    missing_rate : fraction of feature cells to blank out, e.g. 0.1 = 10%.
    random_state : int, for reproducible masking.

    Returns
    -------
    (flawed_df, info) where info is a dict describing what was done.
    """
    rng = np.random.default_rng(random_state)
    flawed_df = df.copy()

    feature_cols = [c for c in df.columns if c != target_col]
    n_rows = len(df)
    n_cols = len(feature_cols)

    mask = rng.random((n_rows, n_cols)) < missing_rate
    feature_values = flawed_df[feature_cols].to_numpy(dtype=float, copy=True)
    feature_values[mask] = np.nan
    flawed_df[feature_cols] = feature_values

    actual_rate = mask.sum() / mask.size

    info = {
        "flaw": "missing_values",
        "target_missing_rate": missing_rate,
        "actual_missing_rate": round(actual_rate, 4),
        "cells_blanked": int(mask.sum()),
    }
    return flawed_df, info