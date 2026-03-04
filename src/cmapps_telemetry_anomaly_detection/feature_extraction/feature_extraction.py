"""
feature_extraction.py

Feature engineering pipeline for CMAPSS FD001 preprocessed data.

Pipeline:
    data/processed/  ->  data/features/

Feature types extracted (per engine, per cycle):
    1. Rolling mean       — smoothed sensor signal (removes noise)
    2. Rolling std        — local variability (rising std = degradation)
    3. Rate of change     — how fast each sensor is trending (diff)
    4. Cumulative mean    — overall engine health trend since cycle 1
    5. Cumulative std     — overall variability since cycle 1

Why these features?
    Raw sensor values alone don't tell you much.
    Anomalies show up as:
        - sudden spikes     → caught by rolling std / rate of change
        - slow drift        → caught by rolling mean / cumulative mean
        - increasing noise  → caught by cumulative std
    By engineering these features, we give models richer signals to learn from,
    improving anomaly detection performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml


# ─────────────────────────────────────────────
# Feature engineering helpers
# ─────────────────────────────────────────────

def rolling_mean(group: pd.DataFrame, sensor_cols: list[str], window: int) -> pd.DataFrame:
    """
    Rolling average over a window of cycles.
    Smooths out short-term noise so models see the underlying trend.

    Example: if sensor_2 spikes for 1 cycle then recovers,
    rolling mean won't flag it. If it stays high, rolling mean rises.
    """
    rolled = (
        group[sensor_cols]
        .rolling(window=window, min_periods=1)
        .mean()
    )
    rolled.columns = [f"{c}_roll_mean_{window}" for c in sensor_cols]
    return rolled


def rolling_std(group: pd.DataFrame, sensor_cols: list[str], window: int) -> pd.DataFrame:
    """
    Rolling standard deviation over a window of cycles.
    Captures local variability — engines near failure often become
    more erratic (rising std is a strong anomaly signal).
    """
    rolled = (
        group[sensor_cols]
        .rolling(window=window, min_periods=1)
        .std()
        .fillna(0)  # first rows have no std yet — fill with 0
    )
    rolled.columns = [f"{c}_roll_std_{window}" for c in sensor_cols]
    return rolled


def rate_of_change(group: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    """
    Cycle-over-cycle difference for each sensor.
    Captures how fast each sensor is changing right now.

    A large positive rate of change in a temperature sensor
    could indicate a fault developing.
    """
    diff = group[sensor_cols].diff().fillna(0)
    diff.columns = [f"{c}_roc" for c in sensor_cols]
    return diff


def cumulative_mean(group: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    """
    Expanding (cumulative) mean from cycle 1 to current cycle.
    Captures the overall health baseline of the engine over its lifetime.

    Unlike rolling mean (local), this tracks long-term drift
    from the engine's own early-life behavior.
    """
    expanded = group[sensor_cols].expanding(min_periods=1).mean()
    expanded.columns = [f"{c}_cum_mean" for c in sensor_cols]
    return expanded


def cumulative_std(group: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    """
    Expanding (cumulative) std from cycle 1 to current cycle.
    An engine with increasing cumulative std is getting noisier
    over its lifetime — a strong degradation signal.
    """
    expanded = (
        group[sensor_cols]
        .expanding(min_periods=1)
        .std()
        .fillna(0)
    )
    expanded.columns = [f"{c}_cum_std" for c in sensor_cols]
    return expanded


# ─────────────────────────────────────────────
# Core extraction logic
# ─────────────────────────────────────────────

def extract_features_for_split(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int,
) -> pd.DataFrame:
    """
    Apply all feature engineering to a full train or test DataFrame.

    Features are computed PER ENGINE (grouped by unit_id) so that
    rolling/cumulative stats don't bleed across different engines.

    Returns the original DataFrame with all new feature columns appended.
    """
    all_features = []

    for unit_id, group in df.groupby("unit_id"):
        group = group.sort_values("cycle").copy()

        feat_parts = [
            group,
            rolling_mean(group, sensor_cols, window),
            rolling_std(group, sensor_cols, window),
            rate_of_change(group, sensor_cols),
            cumulative_mean(group, sensor_cols),
            cumulative_std(group, sensor_cols),
        ]

        engine_features = pd.concat(feat_parts, axis=1)
        all_features.append(engine_features)

    result = pd.concat(all_features, axis=0).reset_index(drop=True)
    return result


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    """
    Return only sensor columns from the DataFrame.
    Excludes metadata, labels, and operational settings.
    """
    exclude = {"unit_id", "cycle", "RUL", "anomaly_label",
               "op_setting_1", "op_setting_2", "op_setting_3"}
    return [c for c in df.columns if c not in exclude and not c.startswith("op_")]


# ─────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────

def run_feature_extraction(config_path: str = "configs/data.yaml") -> None:
    """
    Full feature extraction pipeline entry point.

    Reads processed train + test parquet files,
    engineers all features per engine,
    and saves feature matrices to data/features/.

    Called by:  cmapps-tad features
    """

    # ── Load config ───────────────────────────────────────────────────────────
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    processed_path = Path(config["dataset"]["processed_path"])
    features_path  = Path(config["dataset"]["features_path"])
    subset         = config["dataset"].get("subset", "FD001")
    window         = config["features"].get("rolling_window", 10)

    features_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Extracting features for subset : {subset}")
    print(f"[INFO] Rolling window size            : {window} cycles")

    # ── Load processed data ───────────────────────────────────────────────────
    train_file = processed_path / f"train_{subset}.parquet"
    test_file  = processed_path / f"test_{subset}.parquet"

    for f in [train_file, test_file]:
        if not f.exists():
            raise FileNotFoundError(
                f"[ERROR] Processed file not found: {f}\n"
                f"        Run 'cmapps-tad preprocess' first."
            )

    train_df = pd.read_parquet(train_file)
    test_df  = pd.read_parquet(test_file)

    print(f"[INFO] Loaded train: {train_df.shape} | test: {test_df.shape}")

    # ── Identify sensor columns ───────────────────────────────────────────────
    sensor_cols = get_sensor_columns(train_df)
    print(f"[INFO] Sensors used for feature engineering: {len(sensor_cols)}")
    print(f"       {sensor_cols}")

    # ── Extract features ──────────────────────────────────────────────────────
    print("[INFO] Extracting features for train set...")
    train_features = extract_features_for_split(train_df, sensor_cols, window)

    print("[INFO] Extracting features for test set...")
    test_features  = extract_features_for_split(test_df, sensor_cols, window)

    # ── Save feature matrices ─────────────────────────────────────────────────
    train_out = features_path / f"train_{subset}.parquet"
    test_out  = features_path / f"test_{subset}.parquet"

    train_features.to_parquet(train_out, index=False)
    test_features.to_parquet(test_out,   index=False)

    print(f"[INFO] Saved train features -> {train_out}")
    print(f"[INFO] Saved test features  -> {test_out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    original_sensor_count = len(sensor_cols)
    total_feature_cols    = len(train_features.columns)
    new_feature_cols      = total_feature_cols - len(train_df.columns)

    print("\n── Feature Extraction Summary ───────────────────────────")
    print(f"  Subset              : {subset}")
    print(f"  Rolling window      : {window} cycles")
    print(f"  Original sensors    : {original_sensor_count}")
    print(f"  New feature columns : {new_feature_cols}")
    print(f"  Total columns       : {total_feature_cols}")
    print(f"  Train feature shape : {train_features.shape}")
    print(f"  Test feature shape  : {test_features.shape}")
    print("─────────────────────────────────────────────────────────\n")
    print("[SUCCESS] Feature extraction complete.")