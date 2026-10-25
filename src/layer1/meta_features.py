"""Compute dataset-, model-, and error-level meta-features for one Layer 1 run."""
import numpy as np
import pandas as pd


def dataset_meta_features(df, target_col="target"):
    """Meta-features computed from the data alone, before training anything."""
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    minority_class_share = y.value_counts(normalize=True).min()
    missing_rate = X.isna().mean().mean()

    # Mean absolute correlation between every pair of feature columns.
    corr_matrix = X.corr().to_numpy()
    n = corr_matrix.shape[0]
    off_diagonal = corr_matrix[~np.eye(n, dtype=bool)]
    mean_feature_correlation = np.nanmean(np.abs(off_diagonal)) if n > 1 else 0.0

    # NEW: strongest link between any two feature columns (a copied column shows up here).
    max_pairwise_correlation = np.nanmax(np.abs(off_diagonal)) if n > 1 else 0.0

    # NEW: share of column pairs that are near-copies of each other.
    frac_near_duplicate_pairs = np.nanmean(np.abs(off_diagonal) > 0.95) if n > 1 else 0.0

    # Highest correlation between any single feature and the target.
    target_corrs = X.corrwith(y).abs()
    max_feature_target_correlation = target_corrs.max() if len(target_corrs) > 0 else 0.0

    # NEW: average strength of the link between features and the target.
    mean_feature_target_correlation = target_corrs.mean() if len(target_corrs) > 0 else 0.0

    # Fraction of cells more than 3 standard deviations from their column's mean.
    z_scores = (X - X.mean()) / X.std()
    outlier_score = (z_scores.abs() > 3).to_numpy().mean()

    return {
        "minority_class_share": round(float(minority_class_share), 4),
        "missing_rate": round(float(missing_rate), 4),
        "mean_feature_correlation": round(float(mean_feature_correlation), 4),
        "max_feature_target_correlation": round(float(max_feature_target_correlation), 4),
        "max_pairwise_correlation": round(float(max_pairwise_correlation), 4),
        "frac_near_duplicate_pairs": round(float(frac_near_duplicate_pairs), 4),
        "mean_feature_target_correlation": round(float(mean_feature_target_correlation), 4),
        "outlier_score": round(float(outlier_score), 4),
        "n_rows": len(df),
        "n_features": len(feature_cols),
    }


def model_meta_features(y_train_true, y_train_pred, y_test_true, y_test_pred):
    """Meta-features about how well the model performed."""
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    train_accuracy = accuracy_score(y_train_true, y_train_pred)
    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    macro_f1 = f1_score(y_test_true, y_test_pred, average="macro")

    minority_class = pd.Series(y_test_true).value_counts().idxmin()
    minority_recall = recall_score(y_test_true, y_test_pred, pos_label=minority_class)

    return {
        "accuracy": round(float(test_accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "minority_recall": round(float(minority_recall), 4),
        "train_test_gap": round(float(train_accuracy - test_accuracy), 4),
    }


def error_meta_features(y_test_true, y_test_pred, y_test_proba):
    """Meta-features about the pattern of errors, not just how many."""
    y_test_true = np.asarray(y_test_true)
    y_test_pred = np.asarray(y_test_pred)
    is_wrong = y_test_true != y_test_pred

    error_rates = {}
    for cls in np.unique(y_test_true):
        cls_mask = y_test_true == cls
        error_rates[cls] = is_wrong[cls_mask].mean() if cls_mask.sum() > 0 else 0.0

    # Confidence the model had in its own (wrong) prediction, for wrong rows only.
    if is_wrong.sum() > 0:
        confidence_on_wrong_predictions = y_test_proba[is_wrong].max(axis=1).mean()
    else:
        confidence_on_wrong_predictions = 0.0

    return {
        "error_rate_class_0": round(float(error_rates.get(0, 0.0)), 4),
        "error_rate_class_1": round(float(error_rates.get(1, 0.0)), 4),
        "mean_confidence_on_errors": round(float(confidence_on_wrong_predictions), 4),
    }

def probe_meta_features(df, target_col="target", max_rows=2000, random_state=0):
    """Symptoms found by PROBING the data: small experiments run on the dataset itself."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = pd.factorize(df[target_col], sort=True)[0]

    # Models cannot read NaN, so fill gaps with the column average (all-empty columns become 0).
    X = X.fillna(X.mean(numeric_only=True)).fillna(0.0)

    # Keep the probe fast on big datasets.
    if len(X) > max_rows:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=random_state)

    X_arr = X.to_numpy(dtype=float)

    # ---- Probe 1: out-of-fold predictions, to see how well each row's label is "believed" ----
    n_splits = int(min(5, np.bincount(y).min()))
    if n_splits >= 2:
        rf = RandomForestClassifier(n_estimators=60, random_state=random_state, n_jobs=-1)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        proba = cross_val_predict(rf, X_arr, y, cv=cv, method="predict_proba")
        given_label_prob = proba[np.arange(len(y)), y]
        confident_disagreement_rate = float((given_label_prob < 0.1).mean())
        mean_given_label_prob = float(given_label_prob.mean())
    else:
        confident_disagreement_rate, mean_given_label_prob = 0.0, 1.0

    # ---- Probe 2: does one single feature dominate the model? ----
    rf_full = RandomForestClassifier(n_estimators=60, random_state=random_state, n_jobs=-1).fit(X_arr, y)
    top_feature_importance_share = float(rf_full.feature_importances_.max())

    # ---- Probe 3: shape of each column (robust spread vs normal spread, extreme values) ----
    std = X_arr.std(axis=0)
    ok = std > 1e-12
    if ok.any():
        Xo = X_arr[:, ok]
        med = np.median(Xo, axis=0)
        mad = np.median(np.abs(Xo - med), axis=0) * 1.4826
        has_spread = mad > 1e-12   # skip mostly-constant columns (e.g. mostly zeros)
        if has_spread.any():
            robust_spread_ratio = float(np.median(mad[has_spread] / Xo.std(axis=0)[has_spread]))
        else:
            robust_spread_ratio = 1.0
        rz = np.abs(Xo - med) / np.where(mad > 1e-12, mad, np.nan)
        robust_outlier_share = float(np.nanmean(rz > 6)) if np.isfinite(rz).any() else 0.0
    else:
        robust_spread_ratio, robust_outlier_share = 1.0, 0.0

    return {
        "confident_disagreement_rate": round(confident_disagreement_rate, 4),
        "mean_given_label_prob": round(mean_given_label_prob, 4),
        "top_feature_importance_share": round(top_feature_importance_share, 4),
        "robust_spread_ratio": round(robust_spread_ratio, 4),
        "robust_outlier_share": round(robust_outlier_share, 4),
    }