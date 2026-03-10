"""
pca_reconstruction.py

PCA Reconstruction anomaly detection model for CMAPSS FD001.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
import joblib
import yaml


# ─────────────────────────────────────────────
# Feature column helpers
# ─────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return only the engineered feature columns for model input.
    Excludes metadata, labels, and operational settings.
    """
    exclude = {
        "unit_id", "cycle", "RUL", "anomaly_label",
        "op_setting_1", "op_setting_2", "op_setting_3",
    }
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────────
# PCA core (from-scratch eigendecomposition)
# ─────────────────────────────────────────────

class PCAReconstruction:
    """
    PCA-based anomaly detector using reconstruction error.

    Fits on healthy data only. Anomaly score = reconstruction error.
    Higher error = more anomalous.

    Uses from-scratch eigendecomposition (no sklearn PCA internals)
    so the math is fully transparent and auditable.
    """

    def __init__(self, n_components: int = None, percentile: float = 95.0):
        """
        Parameters:
            n_components : number of principal components to keep.
                           If None, keeps enough to explain 95% of variance.
            percentile   : training error percentile used to set the
                           default anomaly threshold.
                           e.g. 95.0 means top 5% errors = anomaly.
        """
        self.n_components    = n_components
        self.percentile      = percentile

        # Set after fitting
        self._mean           = None
        self._components     = None
        self._eigvals        = None
        self._scaler         = None
        self._threshold      = None
        self._train_errors   = None
        self._explained_var  = None

    def fit(self, X: np.ndarray) -> "PCAReconstruction":
        """
        Fit PCA on healthy training data.

        Steps:
            1. StandardScale the data (important: sensors have different ranges)
            2. Center the data (subtract mean)
            3. Compute covariance matrix
            4. Eigendecompose covariance -> principal components
            5. Keep top-k components (by explained variance)
            6. Compute training reconstruction errors
            7. Set anomaly threshold at configured percentile
        """
        # Step 1: Scale
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)

        # Step 2: Center
        self._mean = np.mean(Xs, axis=0)
        X_centered = Xs - self._mean

        # Step 3: Covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # Step 4: Eigendecomposition (eigh = symmetric matrix, more stable)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort largest to smallest eigenvalue
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Step 5: Determine number of components
        total_var = eigvals.sum()
        cumulative_var = np.cumsum(eigvals) / total_var

        if self.n_components is None:
            # Auto-select: keep components explaining 95% of variance
            self.n_components = int(np.searchsorted(cumulative_var, 0.95)) + 1

        self._components    = eigvecs[:, :self.n_components]
        self._eigvals       = eigvals[:self.n_components]
        self._explained_var = cumulative_var[self.n_components - 1]

        # Step 6: Compute training reconstruction errors
        self._train_errors = self._reconstruction_error(Xs)

        # Step 7: Set threshold
        self._threshold = np.percentile(self._train_errors, self.percentile)

        return self

    def _reconstruction_error(self, Xs: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for scaled input Xs.
        Error = sum of squared differences between original and reconstructed.
        """
        X_centered  = Xs - self._mean
        Z           = X_centered @ self._components          # project to subspace
        X_recon     = (Z @ self._components.T) + self._mean  # project back
        errors      = np.sum((Xs - X_recon) ** 2, axis=1)
        return errors

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute normalized anomaly scores for input X.
        Returns values in [0, 1] where higher = more anomalous.
        Consistent with Isolation Forest output for ensemble compatibility.
        """
        Xs     = self._scaler.transform(X)
        errors = self._reconstruction_error(Xs)

        # Normalize to [0, 1] using training error range
        min_e  = self._train_errors.min()
        max_e  = self._train_errors.max()
        normalized = (errors - min_e) / (max_e - min_e + 1e-9)
        return np.clip(normalized, 0, None)  # scores can exceed 1 for test anomalies

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Binary anomaly predictions using the fitted threshold.
        Returns 1 = anomaly, 0 = normal.
        """
        Xs     = self._scaler.transform(X)
        errors = self._reconstruction_error(Xs)
        return (errors > self._threshold).astype(int)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_pca_reconstruction(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_components: int = None,
    percentile: float = 95.0,
) -> PCAReconstruction:
    """
    Train PCA Reconstruction model on HEALTHY cycles only.

    Same training strategy as Isolation Forest:
        Learn what "normal" looks like, flag deviations.
    """
    healthy_df = train_df[train_df["anomaly_label"] == 0]

    print(f"[INFO] Training on healthy cycles only")
    print(f"[INFO] Healthy rows  : {len(healthy_df):,}  /  Total: {len(train_df):,}")
    print(f"[INFO] Features      : {len(feature_cols)}")
    print(f"[INFO] n_components  : {'auto (95% variance)' if n_components is None else n_components}")
    print(f"[INFO] Threshold pct : {percentile}th percentile of training errors")

    X_train = healthy_df[feature_cols].values

    model = PCAReconstruction(n_components=n_components, percentile=percentile)
    model.fit(X_train)

    print(f"[INFO] Components selected    : {model.n_components}")
    print(f"[INFO] Explained variance     : {model._explained_var:.4f}")
    print(f"[INFO] Anomaly threshold      : {model._threshold:.6f}")
    print("[INFO] PCA Reconstruction training complete.")

    return model


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def score_dataframe(
    model: PCAReconstruction,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Generate anomaly scores and binary predictions for every row.
    Outputs consistent with Isolation Forest for ensemble compatibility.
    """
    X = df[feature_cols].values

    anomaly_scores   = model.score_samples(X)
    predicted_labels = model.predict(X)

    scored_df = df.copy()
    scored_df["anomaly_score"]   = anomaly_scores
    scored_df["predicted_label"] = predicted_labels

    return scored_df


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(scored_df: pd.DataFrame) -> dict:
    """
    Evaluate against proxy anomaly labels.
    Identical metric set to Isolation Forest for direct comparison.
    """
    y_true  = scored_df["anomaly_label"].values
    y_pred  = scored_df["predicted_label"].values
    y_score = scored_df["anomaly_score"].values

    metrics = {
        "precision" : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"    : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1"        : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc"   : round(roc_auc_score(y_true, y_score), 4),
        "pr_auc"    : round(average_precision_score(y_true, y_score), 4),
        "total_rows"          : int(len(scored_df)),
        "true_anomalies"      : int(y_true.sum()),
        "predicted_anomalies" : int(y_pred.sum()),
    }
    return metrics


def print_metrics(metrics: dict, model_name: str = "PCA Reconstruction") -> None:
    print(f"\n── {model_name} Evaluation ──────────────────────────────")
    print(f"  Precision           : {metrics['precision']:.4f}")
    print(f"  Recall              : {metrics['recall']:.4f}")
    print(f"  F1 Score            : {metrics['f1']:.4f}")
    print(f"  ROC-AUC             : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC              : {metrics['pr_auc']:.4f}")
    print(f"  True anomalies      : {metrics['true_anomalies']:,}")
    print(f"  Predicted anomalies : {metrics['predicted_anomalies']:,}")
    print(f"  Total rows scored   : {metrics['total_rows']:,}")
    print("─────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────

def run_pca_reconstruction(config_path: str = "configs/data.yaml") -> None:
    """
    Full PCA Reconstruction train + score + evaluate pipeline.

    Called by:  cmapps-tad train --model pca_reconstruction
    """

    # ── Load config ───────────────────────────────────────────────────────────
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    features_path = Path(config["dataset"]["features_path"])
    subset        = config["dataset"].get("subset", "FD001")

    model_dir     = Path(config["artifacts"]["model_path"])
    metrics_dir   = Path(config["reports"]["metrics_path"])
    tables_dir    = Path(config["reports"]["tables_path"])

    n_components  = config["models"]["pca_reconstruction"].get("n_components", None)
    percentile    = config["models"]["pca_reconstruction"].get("percentile", 95.0)

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ── Load feature data ─────────────────────────────────────────────────────
    train_file = features_path / f"train_{subset}.parquet"
    if not train_file.exists():
        raise FileNotFoundError(
            f"[ERROR] Features not found: {train_file}\n"
            f"        Run 'cmapps-tad features' first."
        )

    print(f"[INFO] Loading features from: {train_file}")
    train_df = pd.read_parquet(train_file)
    feature_cols = get_feature_columns(train_df)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = train_pca_reconstruction(
        train_df,
        feature_cols,
        n_components=n_components,
        percentile=percentile,
    )

    # ── Save model artifact ───────────────────────────────────────────────────
    model_path = model_dir / f"pca_reconstruction_{subset}.joblib"
    joblib.dump(model, model_path)
    print(f"[INFO] Model saved -> {model_path}")

    # ── Score full training set ───────────────────────────────────────────────
    print("[INFO] Scoring full training set...")
    scored_df = score_dataframe(model, train_df, feature_cols)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate(scored_df)
    metrics["model"]  = "pca_reconstruction"
    metrics["subset"] = subset
    metrics["config"] = {
        "n_components": model.n_components,
        "percentile"  : percentile,
        "explained_variance": round(float(model._explained_var), 4),
    }

    print_metrics(metrics)

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_path = metrics_dir / f"pca_reconstruction_{subset}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved -> {metrics_path}")

    # ── Save scores table ─────────────────────────────────────────────────────
    scores_path = tables_dir / f"pca_reconstruction_{subset}_scores.csv"
    scored_df[["unit_id", "cycle", "RUL", "anomaly_label",
               "anomaly_score", "predicted_label"]].to_csv(scores_path, index=False)
    print(f"[INFO] Scores saved -> {scores_path}")

    print("[SUCCESS] PCA Reconstruction pipeline complete.")