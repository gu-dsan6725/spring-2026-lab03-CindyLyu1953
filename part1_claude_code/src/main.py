"""Main pipeline orchestrator for Wine Classification project."""

import logging
import sys

from eda import perform_eda
from evaluate_model import evaluate_model
from generate_report import generate_report
from train_model import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)


def run_pipeline() -> None:
    """Execute the complete Wine Classification ML pipeline."""
    logging.info("=" * 80)
    logging.info("WINE CLASSIFICATION ML PIPELINE")
    logging.info("=" * 80)
    logging.info("")

    try:
        logging.info("STAGE 1/5: Exploratory Data Analysis")
        logging.info("-" * 80)
        perform_eda()
        logging.info("")

        logging.info("STAGE 2/5: Model Training with Hyperparameter Tuning")
        logging.info("-" * 80)
        model, training_metrics = train_model()
        logging.info("")

        logging.info("STAGE 3/5: Model Evaluation")
        logging.info("-" * 80)
        evaluation_metrics = evaluate_model()
        logging.info("")

        logging.info("STAGE 4/5: Report Generation")
        logging.info("-" * 80)
        generate_report()
        logging.info("")

        logging.info("=" * 80)
        logging.info("PIPELINE COMPLETE")
        logging.info("=" * 80)
        logging.info("")
        logging.info("Summary:")
        logging.info(f"  - Test Accuracy: {evaluation_metrics['test_accuracy']:.4f}")
        logging.info(f"  - Test F1-Score (macro): {evaluation_metrics['f1_macro']:.4f}")
        logging.info(
            f"  - Best CV F1-Score: {training_metrics['tuning_results']['best_score']:.4f}"
        )
        logging.info("")
        logging.info("Output files saved to: output/")
        logging.info("  - Plots: class_distribution.png, feature_correlation.png, etc.")
        logging.info("  - Model: trained_model.pkl")
        logging.info("  - Metrics: evaluation_metrics.json, tuning_results.json")
        logging.info("  - Report: wine_classification_report.md")

    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        logging.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
