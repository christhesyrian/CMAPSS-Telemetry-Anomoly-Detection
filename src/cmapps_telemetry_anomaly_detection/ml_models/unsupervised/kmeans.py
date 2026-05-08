"""
kmeans.py

K-Means Clustering anomaly detection model for CMAPSS FD001.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import joblib
import yaml


# ─────────────────────────────────────────────
# Feature column helpers
# ─────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "unit_id", "cycle", "RUL", "anomaly_label",
        "op_setting_1", "op_setting_2", "op_setting_3",
    }
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_kmeans(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 8,
    random_state: int = 42,
) -> tuple[KMeans, StandardScaler]:
    """
    Train K-Means on HEALTHY cycles only.

    Parameters:
        n_clusters   : number of cluster centers to learn.
                       More clusters = finer-grained healthy regions.
                       8 is a solid default for FD001.
        random_state : reproducibility
    """
    healthy_df = train_df[train_df["anomaly_label"] == 0]

    print(f"[INFO] Training on healthy cycles only")
    print(f"[INFO] Healthy rows : {len(healthy_df):,}  /  Total: {len(train_df):,}")
    print(f"[INFO] Features     : {len(feature_cols)}")
    print(f"[INFO] n_clusters   : {n_clusters}")

    X_train = healthy_df[feature_cols].values

    # Scale first — critical for distance-based models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,    # run 10 initialisations to avoid bad local minima
        max_iter=300,
    )
    model.fit(X_scaled)

    print("[INFO] K-Means training complete.")
    return model, scaler


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def score_dataframe(
    model: KMeans,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold_percentile: float = 95.0,
    train_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Score every row by distance to nearest cluster center.

    Threshold is set at the given percentile of TRAINING distances
    so the model flags the top X% most anomalous training points.
    Scores are normalized to [0, 1] for ensemble compatibility.
    """
    X = scaler.transform(df[feature_cols].values)

    # Distance to nearest cluster center
    distances = np.min(
        np.linalg.norm(X[:, np.newaxis] - model.cluster_centers_, axis=2),
        axis=1,
    )

    # Compute threshold from healthy training distances
    if train_df is not None:
        X_train = scaler.transform(
            train_df[train_df["anomaly_label"] == 0][feature_cols].values
        )
        train_distances = np.min(
            np.linalg.norm(X_train[:, np.newaxis] - model.cluster_centers_, axis=2),
            axis=1,
        )
        threshold = np.percentile(train_distances, threshold_percentile)
        min_d, max_d = train_distances.min(), train_distances.max()
    else:
        threshold = np.percentile(distances, threshold_percentile)
        min_d, max_d = distances.min(), distances.max()

    # Normalize scores to [0, 1]
    normalized = (distances - min_d) / (max_d - min_d + 1e-9)

    scored_df = df.copy()
    scored_df["anomaly_score"]   = np.clip(normalized, 0, None)
    scored_df["predicted_label"] = (distances > threshold).astype(int)

    return scored_df


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(scored_df: pd.DataFrame) -> dict:
    y_true  = scored_df["anomaly_label"].values
    y_pred  = scored_df["predicted_label"].values
    y_score = scored_df["anomaly_score"].values

    return {
        "precision" : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"    : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1"        : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc"   : round(roc_auc_score(y_true, y_score), 4),
        "pr_auc"    : round(average_precision_score(y_true, y_score), 4),
        "total_rows"          : int(len(scored_df)),
        "true_anomalies"      : int(y_true.sum()),
        "predicted_anomalies" : int(y_pred.sum()),
    }


def print_metrics(metrics: dict, model_name: str = "K-Means") -> None:
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

def run_kmeans(config_path: str = "configs/data.yaml") -> None:
    """
    Full K-Means train + score + evaluate pipeline.
    Called by:  cmapps-tad train --model kmeans
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    features_path = Path(config["dataset"]["features_path"])
    subset        = config["dataset"].get("subset", "FD001")
    model_dir     = Path(config["artifacts"]["model_path"])
    metrics_dir   = Path(config["reports"]["metrics_path"])
    tables_dir    = Path(config["reports"]["tables_path"])

    n_clusters    = config["models"]["kmeans"].get("n_clusters", 8)
    threshold_pct = config["models"]["kmeans"].get("threshold_percentile", 95.0)
    random_state  = config["models"].get("random_state", 42)

    for d in [model_dir, metrics_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_file = features_path / f"train_{subset}.parquet"
    if not train_file.exists():
        raise FileNotFoundError(f"[ERROR] Run 'cmapps-tad features' first.")

    print(f"[INFO] Loading features from: {train_file}")
    train_df     = pd.read_parquet(train_file)
    feature_cols = get_feature_columns(train_df)

    # Train
    model, scaler = train_kmeans(train_df, feature_cols, n_clusters, random_state)

    # Save model + scaler together
    artifact = {"model": model, "scaler": scaler}
    model_path = model_dir / f"kmeans_{subset}.joblib"
    joblib.dump(artifact, model_path)
    print(f"[INFO] Model saved -> {model_path}")

    # Score
    print("[INFO] Scoring full training set...")
    scored_df = score_dataframe(model, scaler, train_df, feature_cols,
                                threshold_pct, train_df)

    # Evaluate
    metrics = evaluate(scored_df)
    metrics["model"]  = "kmeans"
    metrics["subset"] = subset
    metrics["config"] = {"n_clusters": n_clusters, "threshold_percentile": threshold_pct}
    print_metrics(metrics)

    # Save outputs
    metrics_path = metrics_dir / f"kmeans_{subset}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved -> {metrics_path}")

    scores_path = tables_dir / f"kmeans_{subset}_scores.csv"
    scored_df[["unit_id", "cycle", "RUL", "anomaly_label",
               "anomaly_score", "predicted_label"]].to_csv(scores_path, index=False)
    print(f"[INFO] Scores saved -> {scores_path}")
    print("[SUCCESS] K-Means pipeline complete.")