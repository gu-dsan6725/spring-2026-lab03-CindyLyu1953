"""Exploratory Data Analysis for Wine Classification Dataset."""

import json
import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: Path = Path("output")
RANDOM_STATE: int = 42
IQR_MULTIPLIER: float = 1.5


def _detect_outliers_iqr(
    df: pl.DataFrame,
    feature: str,
) -> Tuple[int, float]:
    """Detect outliers using IQR method for a single feature.

    Args:
        df: Polars DataFrame
        feature: Feature name to check for outliers

    Returns:
        Tuple of (number of outliers, percentage of outliers)
    """
    values = df[feature].to_numpy()
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - IQR_MULTIPLIER * iqr
    upper_bound = q3 + IQR_MULTIPLIER * iqr

    outliers = (values < lower_bound) | (values > upper_bound)
    num_outliers = outliers.sum()
    pct_outliers = (num_outliers / len(values)) * 100

    return int(num_outliers), float(pct_outliers)


def _plot_class_distribution(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Plot class distribution bar chart.

    Args:
        df: Polars DataFrame with 'target' column
        output_path: Path to save the plot
    """
    class_counts = df["target"].value_counts().sort("target")

    plt.figure(figsize=(8, 6))
    plt.bar(
        class_counts["target"].to_list(),
        class_counts["count"].to_list(),
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    plt.xlabel("Wine Class", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title("Class Distribution in Wine Dataset", fontsize=14, fontweight="bold")
    plt.xticks([0, 1, 2])
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Class distribution plot saved to {output_path}")


def _plot_correlation_heatmap(
    df: pl.DataFrame,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Plot correlation heatmap for features.

    Args:
        df: Polars DataFrame
        feature_names: List of feature column names
        output_path: Path to save the plot
    """
    corr_matrix = df[feature_names].to_pandas().corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Correlation heatmap saved to {output_path}")


def _plot_feature_distributions(
    df: pl.DataFrame,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Plot distribution histograms for all features.

    Args:
        df: Polars DataFrame
        feature_names: List of feature column names
        output_path: Path to save the plot
    """
    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_names):
        axes[idx].hist(
            df[feature].to_numpy(), bins=30, color="#1f77b4", alpha=0.7, edgecolor="black"
        )
        axes[idx].set_title(feature, fontsize=10, fontweight="bold")
        axes[idx].set_xlabel("Value", fontsize=9)
        axes[idx].set_ylabel("Frequency", fontsize=9)
        axes[idx].grid(axis="y", alpha=0.3)

    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle("Feature Distributions", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Feature distributions plot saved to {output_path}")


def _plot_feature_by_class(
    df: pl.DataFrame,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Plot box plots of features by class.

    Args:
        df: Polars DataFrame
        feature_names: List of feature column names
        output_path: Path to save the plot
    """
    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    df_pandas = df.to_pandas()

    for idx, feature in enumerate(feature_names):
        sns.boxplot(data=df_pandas, x="target", y=feature, ax=axes[idx], palette="Set2")
        axes[idx].set_title(feature, fontsize=10, fontweight="bold")
        axes[idx].set_xlabel("Wine Class", fontsize=9)
        axes[idx].set_ylabel("Value", fontsize=9)
        axes[idx].grid(axis="y", alpha=0.3)

    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle("Feature Distributions by Wine Class", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Feature by class plot saved to {output_path}")


def perform_eda() -> None:
    """Perform comprehensive exploratory data analysis on Wine dataset."""
    logging.info("Starting exploratory data analysis for Wine dataset")

    OUTPUT_DIR.mkdir(exist_ok=True)

    wine_data = load_wine()
    logging.info(
        f"Loaded Wine dataset with {wine_data.data.shape[0]} samples and {wine_data.data.shape[1]} features"
    )

    df = pl.DataFrame(wine_data.data, schema=wine_data.feature_names)
    df = df.with_columns(pl.Series("target", wine_data.target))

    logging.info("=" * 80)
    logging.info("SUMMARY STATISTICS")
    logging.info("=" * 80)

    logging.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logging.info(f"Features: {', '.join(wine_data.feature_names)}")
    logging.info(f"Target classes: {wine_data.target_names.tolist()}")

    stats = df.select(wine_data.feature_names).describe()
    logging.info(f"\nDescriptive statistics:\n{stats}")

    missing_values = df.null_count()
    logging.info(
        f"Missing values check: {missing_values.sum_horizontal().to_list()[0]} total missing values"
    )

    logging.info("=" * 80)
    logging.info("CLASS BALANCE CHECK")
    logging.info("=" * 80)

    class_counts = df["target"].value_counts().sort("target")
    total_samples = df.shape[0]

    class_balance = {}
    for row in class_counts.iter_rows():
        class_id = int(row[0])
        count = int(row[1])
        percentage = (count / total_samples) * 100
        class_balance[f"Class_{class_id}"] = {
            "count": count,
            "percentage": round(percentage, 2),
            "name": wine_data.target_names[class_id],
        }

    logging.info(f"Class balance:\n{json.dumps(class_balance, indent=2)}")

    logging.info("=" * 80)
    logging.info("OUTLIER DETECTION (IQR Method)")
    logging.info("=" * 80)

    outlier_summary = {}
    for feature in wine_data.feature_names:
        num_outliers, pct_outliers = _detect_outliers_iqr(df, feature)
        outlier_summary[feature] = {
            "num_outliers": num_outliers,
            "pct_outliers": round(pct_outliers, 2),
        }
        if num_outliers > 0:
            logging.info(f"{feature}: {num_outliers} outliers ({pct_outliers:.2f}%)")

    logging.info("=" * 80)
    logging.info("GENERATING VISUALIZATIONS")
    logging.info("=" * 80)

    _plot_class_distribution(df, OUTPUT_DIR / "class_distribution.png")

    _plot_correlation_heatmap(df, wine_data.feature_names, OUTPUT_DIR / "feature_correlation.png")

    _plot_feature_distributions(
        df, wine_data.feature_names, OUTPUT_DIR / "feature_distributions.png"
    )

    _plot_feature_by_class(df, wine_data.feature_names, OUTPUT_DIR / "features_by_class.png")

    logging.info("=" * 80)
    logging.info("EDA COMPLETE")
    logging.info("=" * 80)
    logging.info(f"All plots saved to {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    perform_eda()
