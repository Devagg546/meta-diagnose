"""Inject a data-leakage feature into a clean dataset."""
import numpy as np
import pandas as pd


def inject_leakage(df, target_col="target", leak_strength=0.9, random_state=None):
    """
    Return a copy of df with one extra feature column that is partly
    derived from the target itself, simulating an accidental leak.

    Parameters
    ----------
    df : DataFrame with a binary target column.
    target_col : name of the target column.
    leak_strength : how strongly the new column reveals the target,
        from 0 (pure noise, no leak) to 1 (perfectly reveals the target).
    random_state : int, for reproducible noise.

    Returns
    -------
    (flawed_df, info) where info is a dict describing what was done.
    """
    rng = np.random.default_rng(random_state)
    flawed_df = df.copy()

    target = df[target_col].to_numpy(dtype=float)
    noise = rng.normal(loc=0.0, scale=1.0, size=len(df))

    # Blend the true target with random noise. At leak_strength=1, the
    # column is just the target (perfect leak). At 0, it's pure noise.
    leaked_col = leak_strength * target + (1 - leak_strength) * noise

    col_name = "leaked_feature"
    flawed_df[col_name] = leaked_col

    correlation = np.corrcoef(leaked_col, target)[0, 1]

    info = {
        "flaw": "leakage",
        "leak_strength": leak_strength,
        "leaked_column": col_name,
        "actual_correlation_with_target": round(float(correlation), 4),
    }
    return flawed_df, info