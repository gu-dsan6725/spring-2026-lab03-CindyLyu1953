import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
OUTPUT_DIR: Path = Path("output")
PLOTS_DIR: Path = OUTPUT_DIR / "plots"
SUMMARY_FILE: Path = OUTPUT_DIR / "eda_summary.txt"
FIG_SIZE: tuple[int, int] = (10, 8)
CORR_FIG_SIZE: tuple[int, int] = (12, 10)


def _setup_directories() -> None:
    """Ensure output directories exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> pl.DataFrame:
    """Load the Wine dataset and convert to Polars DataFrame."""
    logging.info("Loading Wine dataset...")
    data: Any = load_wine()
    df: pl.DataFrame = pl.DataFrame(data.data)
    df.columns = data.feature_names
    df = df.with_columns(pl.Series("target", data.target))
    logging.info(f"Dataset loaded with shape: {df.shape}")
    return df


def _save_summary_statistics(df: pl.DataFrame) -> None:
    """Compute and save summary statistics."""
    logging.info("Computing summary statistics...")
    summary: pl.DataFrame = df.describe()

    with open(SUMMARY_FILE, "w") as f:
        f.write("Wine Dataset Summary Statistics\n")
        f.write("===============================\n\n")
        f.write(str(summary))

    logging.info(f"Summary statistics saved to {SUMMARY_FILE}")


def _plot_distributions(df: pl.DataFrame) -> None:
    """Generate and save distribution plots for all features."""
    logging.info("Generating distribution plots...")
    features: list[str] = [col for col in df.columns if col != "target"]

    for feature in features:
        plt.figure(figsize=FIG_SIZE)
        sns.histplot(data=df, x=feature, hue="target", kde=True, palette="viridis")
        plt.title(f"Distribution of {feature}")
        plt.tight_layout()

        # Sanitize filename
        safe_feature_name = feature.replace("/", "_").replace(" ", "_")
        output_path: Path = PLOTS_DIR / f"dist_{safe_feature_name}.png"

        plt.savefig(output_path)
        plt.close()

    logging.info(f"Distribution plots saved to {PLOTS_DIR}")


def _plot_correlation_heatmap(df: pl.DataFrame) -> None:
    """Generate and save correlation heatmap."""
    logging.info("Generating correlation heatmap...")
    corr_matrix: pl.DataFrame = df.corr()
    # Convert to pandas for seaborn compatibility (heatmap requires matrix/dataframe)
    # polars -> numpy/pandas for seaborn
    corr_pd = corr_matrix.to_pandas()
    # Set index to columns for proper labeling in heatmap
    corr_pd.index = df.columns
    corr_pd.columns = df.columns

    plt.figure(figsize=CORR_FIG_SIZE)
    sns.heatmap(corr_pd, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    output_path: Path = PLOTS_DIR / "correlation_heatmap.png"
    plt.savefig(output_path)
    plt.close()

    logging.info(f"Correlation heatmap saved to {output_path}")


def _check_class_balance(df: pl.DataFrame) -> None:
    """Check and visualize class balance."""
    logging.info("Checking class balance...")
    class_counts: pl.DataFrame = df["target"].value_counts()
    logging.info(f"Class counts:\n{class_counts}")

    plt.figure(figsize=FIG_SIZE)
    sns.barplot(x=class_counts["target"], y=class_counts["count"], palette="viridis")
    plt.title("Class Distribution")
    plt.xlabel("Target Class")
    plt.ylabel("Count")
    plt.tight_layout()
    output_path: Path = PLOTS_DIR / "class_balance.png"
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    """Main execution flow for EDA."""
    try:
        _setup_directories()
        df: pl.DataFrame = _load_data()

        _save_summary_statistics(df)
        _plot_distributions(df)
        _plot_correlation_heatmap(df)
        _check_class_balance(df)

        logging.info("EDA completed successfully.")

    except Exception as e:
        logging.error(f"EDA failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
