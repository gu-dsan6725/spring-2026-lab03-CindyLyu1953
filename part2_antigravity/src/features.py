import logging
from typing import Tuple

import polars as pl
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2


def load_and_preprocess_data() -> (
    Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series, list[str]]
):
    """
    Load Wine dataset, create derived features, scale, and split.

    Returns:
        X_train (pl.DataFrame): Training features.
        X_test (pl.DataFrame): Test features.
        y_train (pl.Series): Training labels.
        y_test (pl.Series): Test labels.
        feature_names (list[str]): List of feature names properly ordered.
    """
    logging.info("Loading and preprocessing data...")

    # Load data
    data = load_wine()
    df = pl.DataFrame(data.data)
    df.columns = data.feature_names
    y = pl.Series("target", data.target)

    # Feature Engineering
    logging.info("Creating derived features...")

    # 1. Proline / Magnesium ratio
    df = df.with_columns(
        (pl.col("proline") / pl.col("magnesium")).alias("proline_magnesium_ratio")
    )

    # 2. Alcohol / Ash ratio
    df = df.with_columns((pl.col("alcohol") / pl.col("ash")).alias("alcohol_ash_ratio"))

    # 3. Color Intensity * Hue interaction
    df = df.with_columns(
        (pl.col("color_intensity") * pl.col("hue")).alias("color_hue_product")
    )

    feature_names = df.columns
    logging.info(f"Features after engineering: {len(feature_names)}")

    # Split data (Stratified)
    logging.info("Splitting data into train/test sets...")

    X = df.to_numpy()
    y_np = y.to_numpy()

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X, y_np, test_size=TEST_SIZE, stratify=y_np, random_state=RANDOM_STATE
    )

    # Scaling
    logging.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled = scaler.transform(X_test_np)

    # Convert back to Polars
    X_train = pl.DataFrame(X_train_scaled, schema=feature_names)
    X_test = pl.DataFrame(X_test_scaled, schema=feature_names)
    y_train = pl.Series("target", y_train_np)
    y_test = pl.Series("target", y_test_np)

    logging.info(
        f"Data preprocessing complete. "
        f"Train shape: {X_train.shape}, Test shape: {X_test.shape}"
    )

    return X_train, X_test, y_train, y_test, feature_names
