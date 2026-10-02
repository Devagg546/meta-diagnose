"""Defines all experiment conditions: clean + 7 flaws x 4 severities each = 29 conditions."""

from src.flaws.imbalance import inject_imbalance
from src.flaws.missing import inject_missing
from src.flaws.leakage import inject_leakage
from src.flaws.noise import inject_feature_noise, inject_label_noise
from src.flaws.outliers import inject_outliers
from src.flaws.redundancy import inject_redundancy


FLAW_FUNCTIONS = {
    "imbalance": inject_imbalance,
    "missing_values": inject_missing,
    "leakage": inject_leakage,
    "feature_noise": inject_feature_noise,
    "label_noise": inject_label_noise,
    "outliers": inject_outliers,
    "redundancy": inject_redundancy,
}

SEVERITY_LEVELS = {
    "imbalance":      [{"minority_share": s}   for s in [0.3, 0.15, 0.1, 0.02]],
    "missing_values": [{"missing_rate": s}     for s in [0.05, 0.15, 0.25, 0.4]],
    "leakage":        [{"leak_strength": s}    for s in [0.3, 0.6, 0.8, 0.95]],
    "feature_noise":  [{"noise_level": s}      for s in [0.05, 0.15, 0.3, 0.5]],
    "label_noise":    [{"flip_rate": s}        for s in [0.02, 0.05, 0.1, 0.2]],
    "outliers":       [{"outlier_fraction": s} for s in [0.01, 0.03, 0.05, 0.1]],
    "redundancy":     [{"n_redundant": s}      for s in [1, 2, 3, 5]],
}


def build_condition_list():
    """
    Build the full list of 29 conditions: 1 clean + 7 flaws x 4 severities.

    Returns
    -------
    list of dicts, each: {"condition_id": str, "flaw": str, "params": dict}
    """
    conditions = [{"condition_id": "clean", "flaw": "clean", "params": {}}]

    for flaw_name, severity_list in SEVERITY_LEVELS.items():
        for params in severity_list:
            param_str = "_".join(f"{k}_{v}" for k, v in params.items())
            condition_id = f"{flaw_name}__{param_str}"
            conditions.append({
                "condition_id": condition_id,
                "flaw": flaw_name,
                "params": params,
            })

    return conditions


def apply_condition(df, condition, random_state=None):
    """
    Apply one condition (from build_condition_list) to a clean dataframe.

    Returns
    -------
    (flawed_df, info) — info is whatever the underlying flaw function
    returns, or a minimal dict for the "clean" condition.
    """
    if condition["flaw"] == "clean":
        return df.copy(), {"flaw": "clean"}

    func = FLAW_FUNCTIONS[condition["flaw"]]
    return func(df, random_state=random_state, **condition["params"])