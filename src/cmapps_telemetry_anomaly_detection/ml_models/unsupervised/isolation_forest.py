"""
isolation_forest.py

Isolation Forest anomaly detection model for CMAPSS FD001.

Training strategy:
    - Train ONLY on healthy cycles (anomaly_label == 0, RUL > threshold)
    - Should learn the "normal" behavior of engines so that deviations get high anomaly scores.
    - At scoring time, anything that deviates from normal gets a high anomaly score and is flagged as anomalous.


"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
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
    """
    Return only the engineered feature columns for model input.

    Excludes:
        - metadata  : unit_id, cycle
        - labels    : RUL, anomaly_label
        - op settings: op_setting_1/2/3

    Raw sensors + engineered features is used.
    """
    exclude = {
        "unit_id", "cycle", "RUL", "anomaly_label",
        "op_setting_1", "op_setting_2", "op_setting_3",
    }
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_isolation_forest(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_estimators: int = 100,
    contamination: float = 0.05,
    random_state: int = 42,
) -> IsolationForest:
    """
    Train Isolation Forest on HEALTHY cycles only.

    Parameters:
        n_estimators   : number of trees (more = more stable, slower)
        contamination  : expected proportion of anomalies in scoring data
                         (used to set the decision threshold internally)
        random_state   : for reproducibility
    """
    # Filter to healthy cycles only
    healthy_df = train_df[train_df["anomaly_label"] == 0]

    print(f"[INFO] Training on healthy cycles only")
    print(f"[INFO] Healthy rows : {len(healthy_df):,}  /  Total rows: {len(train_df):,}")
    print(f"[INFO] Features     : {len(feature_cols)}")
    print(f"[INFO] n_estimators : {n_estimators} | contamination: {contamination}")

    X_train = healthy_df[feature_cols].values

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,  # use all CPU cores
    )
    model.fit(X_train)

    print("[INFO] Isolation Forest training complete.")
    return model


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def score_dataframe(
    model: IsolationForest,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Generate anomaly scores for every row in the DataFrame.

    Isolation Forest outputs:
        decision_function score : more negative = more anomalous
        binary prediction       : -1 = anomaly, 1 = normal

    We convert to:
        anomaly_score : higher = more anomalous  (flipped + normalized to [0, 1])
        predicted_label : 1 = anomaly, 0 = normal  (converted from -1/1)
    """
    X = df[feature_cols].values

    raw_scores = model.decision_function(X)     # more negative = more anomalous
    predictions = model.predict(X)              # -1 = anomaly, 1 = normal

    # Flip so higher = more anomalous, then normalize to [0, 1]
    flipped = -raw_scores
    normalized = (flipped - flipped.min()) / (flipped.max() - flipped.min() + 1e-9)

    # Convert -1/1 predictions to 1/0
    binary_preds = (predictions == -1).astype(int)

    scored_df = df.copy()
    scored_df["anomaly_score"]   = normalized
    scored_df["predicted_label"] = binary_preds

    return scored_df


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(scored_df: pd.DataFrame) -> dict:
    """
    Evaluate anomaly detection performance against proxy labels.

    Metrics:
        Precision   : of all rows flagged as anomalous, how many truly are?
        Recall      : of all truly anomalous rows, how many did we catch?
        F1          : harmonic mean of precision + recall
        ROC-AUC     : area under ROC curve (threshold-independent)
        PR-AUC      : area under Precision-Recall curve
                      (better than ROC-AUC for imbalanced datasets like this)
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


def print_metrics(metrics: dict, model_name: str = "Isolation Forest") -> None:
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

def run_isolation_forest(config_path: str = "configs/data.yaml") -> None:
    """
    Full Isolation Forest train + score + evaluate pipeline.

    Called by:  cmapps-tad train --model isolation_forest
    """

    # ── Load config ───────────────────────────────────────────────────────────
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    features_path  = Path(config["dataset"]["features_path"])
    subset         = config["dataset"].get("subset", "FD001")

    model_dir      = Path(config["artifacts"]["model_path"])
    metrics_dir    = Path(config["reports"]["metrics_path"])
    tables_dir     = Path(config["reports"]["tables_path"])

    n_estimators   = config["models"]["isolation_forest"].get("n_estimators", 100)
    contamination  = config["models"]["isolation_forest"].get("contamination", 0.05)
    random_state   = config["models"].get("random_state", 42)

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
    model = train_isolation_forest(
        train_df,
        feature_cols,
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )

    # ── Save model artifact ───────────────────────────────────────────────────
    model_path = model_dir / f"isolation_forest_{subset}.joblib"
    joblib.dump(model, model_path)
    print(f"[INFO] Model saved -> {model_path}")

    # ── Score full training set ───────────────────────────────────────────────
    print("[INFO] Scoring full training set...")
    scored_df = score_dataframe(model, train_df, feature_cols)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate(scored_df)
    metrics["model"]  = "isolation_forest"
    metrics["subset"] = subset
    metrics["config"] = {
        "n_estimators" : n_estimators,
        "contamination": contamination,
        "random_state" : random_state,
    }

    print_metrics(metrics)

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_path = metrics_dir / f"isolation_forest_{subset}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved -> {metrics_path}")

    # ── Save scores table ─────────────────────────────────────────────────────
    scores_path = tables_dir / f"isolation_forest_{subset}_scores.csv"
    scored_df[["unit_id", "cycle", "RUL", "anomaly_label",
               "anomaly_score", "predicted_label"]].to_csv(scores_path, index=False)
    print(f"[INFO] Scores saved -> {scores_path}")

    print("[SUCCESS] Isolation Forest pipeline complete.")