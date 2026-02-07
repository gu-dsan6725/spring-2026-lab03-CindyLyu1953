"""Feature engineering and data preparation for Wine Classification."""

import logging
from typing import Tuple

import numpy as np
import polars as pl
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2


def _create_ratio_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Create ratio-based derived features.

    Args:
        df: Input polars DataFrame with wine features

    Returns:
        DataFrame with added ratio features
    """
    df = df.with_columns(
        (pl.col("flavanoids") / (pl.col("total_phenols") + 1e-8)).alias("flavanoids_phenols_ratio")
    )

    df = df.with_columns(
        (pl.col("od280/od315_of_diluted_wines") / (pl.col("proline") + 1e-8)).alias(
            "od_proline_ratio"
        )
    )

    df = df.with_columns(
        (pl.col("malic_acid") / (pl.col("alcohol") + 1e-8)).alias("malic_alcohol_ratio")
    )

    return df


def _create_interaction_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Create interaction features by multiplying related features.

    Args:
        df: Input polars DataFrame with wine features

    Returns:
        DataFrame with added interaction features
    """
    df = df.with_columns(
        (pl.col("alcohol") * pl.col("flavanoids")).alias("alcohol_flavanoids_interaction")
    )

    df = df.with_columns((pl.col("color_intensity") * pl.col("hue")).alias("color_hue_interaction"))

    return df


def prepare_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Prepare features with engineering, scaling, and train/test split.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    logging.info("Starting feature engineering and data preparation")

    wine_data = load_wine()
    logging.info(f"Loaded Wine dataset: {wine_data.data.shape[0]} samples")

    df = pl.DataFrame(wine_data.data, schema=wine_data.feature_names)
    df = df.with_columns(pl.Series("target", wine_data.target))

    logging.info("Creating derived features...")

    df = _create_ratio_features(df)
    logging.info(
        "Created 3 ratio features: flavanoids_phenols_ratio, od_proline_ratio, malic_alcohol_ratio"
    )

    df = _create_interaction_features(df)
    logging.info(
        "Created 2 interaction features: alcohol_flavanoids_interaction, color_hue_interaction"
    )

    feature_columns = [col for col in df.columns if col != "target"]
    logging.info(f"Total features after engineering: {len(feature_columns)}")

    X = df.select(feature_columns).to_numpy()
    y = df["target"].to_numpy()

    logging.info(f"Performing stratified train/test split (test_size={TEST_SIZE})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    logging.info(f"Train set: {X_train.shape[0]} samples")
    logging.info(f"Test set: {X_test.shape[0]} samples")

    train_class_dist = np.bincount(y_train)
    test_class_dist = np.bincount(y_test)
    logging.info(f"Train class distribution: {train_class_dist.tolist()}")
    logging.info(f"Test class distribution: {test_class_dist.tolist()}")

    logging.info("Applying standard scaling to features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logging.info("Feature scaling complete")
    logging.info(f"Feature means (after scaling): {X_train_scaled.mean(axis=0)[:5].tolist()}")
    logging.info(f"Feature stds (after scaling): {X_train_scaled.std(axis=0)[:5].tolist()}")

    logging.info("Feature engineering and preparation complete")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = prepare_features()
    logging.info(f"Final shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    logging.info(f"Feature names: {features}")
