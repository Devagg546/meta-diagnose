"""Inject feature noise and label noise into a clean dataset."""
import numpy as np
import pandas as pd


def inject_feature_noise(df, target_col="target", noise_level=0.1, random_state=None):
    """
    Add Gaussian noise to every feature column, scaled to each column's
    own standard deviation so noise_level is comparable across columns
    with very different ranges.

    Parameters
    ----------
    df : DataFrame with a binary target column.
    target_col : name of the target column, left untouched.
    noise_level : size of the noise relative to each column's spread,
        e.g. 0.1 means noise with a standard deviation of 10% of the
        column's own standard deviation.
    random_state : int, for reproducible noise.

    Returns
    -------
    (flawed_df, info) where info is a dict describing what was done.
    """
    rng = np.random.default_rng(random_state)
    flawed_df = df.copy()

    feature_cols = [c for c in df.columns if c != target_col]

    for col in feature_cols:
        col_std = df[col].std()
        noise = rng.normal(loc=0.0, scale=noise_level * col_std, size=len(df))
        flawed_df[col] = df[col] + noise

    info = {
        "flaw": "feature_noise",
        "noise_level": noise_level,
        "columns_affected": len(feature_cols),
    }
    return flawed_df, info


def inject_label_noise(df, target_col="target", flip_rate=0.05, random_state=None):
    """
    Randomly flip a fraction of target labels (0 <-> 1), simulating
    mislabeled data.

    Parameters
    ----------
    df : DataFrame with a binary target column.
    target_col : name of the target column.
    flip_rate : fraction of rows whose label gets flipped.
    random_state : int, for reproducible flipping.

    Returns
    -------
    (flawed_df, info) where info is a dict describing what was done.
    """
    rng = np.random.default_rng(random_state)
    flawed_df = df.copy()

    n_rows = len(df)
    n_flip = int(round(flip_rate * n_rows))
    flip_indices = rng.choice(n_rows, size=n_flip, replace=False)

    flawed_df.loc[flawed_df.index[flip_indices], target_col] = (
        1 - flawed_df.loc[flawed_df.index[flip_indices], target_col]
    )

    info = {
        "flaw": "label_noise",
        "target_flip_rate": flip_rate,
        "labels_flipped": n_flip,
    }
    return flawed_df, info