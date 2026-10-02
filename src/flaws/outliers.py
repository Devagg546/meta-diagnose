"""Inject extreme outlier values into a clean dataset."""
import numpy as np
import pandas as pd


def inject_outliers(df, target_col="target", outlier_fraction=0.02, scale=8.0, random_state=None):
    """
    Pick a random fraction of cells in the feature columns and replace
    each one with an extreme value: the column's mean plus or minus
    `scale` times its standard deviation.

    Parameters
    ----------
    df : DataFrame with a binary target column.
    target_col : name of the target column, left untouched.
    outlier_fraction : fraction of feature cells to turn into outliers.
    scale : how many standard deviations away from the mean each
        outlier is placed, e.g. 8.0 = 8 standard deviations out.
    random_state : int, for reproducible selection.

    Returns
    -------
    (flawed_df, info) where info is a dict describing what was done.
    """
    rng = np.random.default_rng(random_state)
    flawed_df = df.copy()

    feature_cols = [c for c in df.columns if c != target_col]
    n_rows = len(df)
    n_cols = len(feature_cols)

    mask = rng.random((n_rows, n_cols)) < outlier_fraction
    feature_values = flawed_df[feature_cols].to_numpy(dtype=float, copy=True)

    col_means = df[feature_cols].mean().to_numpy()
    col_stds = df[feature_cols].std().to_numpy()

    # Random sign, so outliers land on both the high and low side.
    signs = rng.choice([-1.0, 1.0], size=(n_rows, n_cols))
    extreme_values = col_means + signs * scale * col_stds

    feature_values[mask] = extreme_values[mask]
    flawed_df[feature_cols] = feature_values

    actual_fraction = mask.sum() / mask.size

    info = {
        "flaw": "outliers",
        "target_outlier_fraction": outlier_fraction,
        "actual_outlier_fraction": round(actual_fraction, 4),
        "scale": scale,
        "cells_changed": int(mask.sum()),
    }
    return flawed_df, info