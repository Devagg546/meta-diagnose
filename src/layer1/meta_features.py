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