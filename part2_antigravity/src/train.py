import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

try:
    from src.features import load_and_preprocess_data
except ImportError:
    from features import load_and_preprocess_data

# Constants
OUTPUT_DIR: Path = Path("output")
PLOTS_DIR: Path = OUTPUT_DIR / "plots"
REPORT_FILE: Path = OUTPUT_DIR / "evaluation_report.md"
FIG_SIZE: tuple[int, int] = (10, 8)
N_ITER: int = 20
CV_FOLDS: int = 5
RANDOM_STATE: int = 42


def _setup_logging(debug: bool) -> None:
    """Configure logging based on debug flag."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
        force=True,
    )


def _optimize_hyperparameters(
    X_train: pl.DataFrame, y_train: pl.Series
) -> XGBClassifier:
    """Perform RandomizedSearchCV to find best hyperparameters."""
    logging.info("Starting hyperparameter optimization...")

    param_dist: dict[str, Any] = {
        "n_estimators": [100, 200, 300, 400, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3, 0.4],
    }

    xgb = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
    )

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Convert Polars to pandas for sklearn compatibility
    # Scikit-learn CV splitters work best with pandas or numpy
    X_pd = X_train.to_pandas()
    y_pd = y_train.to_pandas()

    random_search.fit(X_pd, y_pd)

    logging.info(f"Best params: {json.dumps(random_search.best_params_, indent=2)}")
    return random_search.best_estimator_


def _plot_confusion_matrix(cm: Any) -> None:
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=FIG_SIZE)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    output_path = PLOTS_DIR / "confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Confusion matrix saved to {output_path}")


def _plot_feature_importance(importance: Any, feature_names: list[str]) -> None:
    """Plot and save feature importance."""
    fi_df = pl.DataFrame({"feature": feature_names, "importance": importance})
    fi_df = fi_df.sort("importance", descending=True)

    plt.figure(figsize=FIG_SIZE)
    sns.barplot(
        x=fi_df["importance"],
        y=fi_df["feature"],
        palette="viridis",
        hue=fi_df["feature"],
        legend=False,
    )
    plt.title("Feature Importance")
    plt.tight_layout()
    output_path = PLOTS_DIR / "feature_importance.png"
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Feature importance saved to {output_path}")


def _evaluate_model(
    model: XGBClassifier,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    feature_names: list[str],
) -> dict[str, Any]:
    """Evaluate the model and generate metrics."""
    logging.info("Evaluating model...")
    # Convert to pandas/numpy for consistent prediction if model was trained on pandas
    X_test_pd = X_test.to_pandas()
    y_test_pd = y_test.to_pandas()

    y_pred = model.predict(X_test_pd)

    # Calculate metrics
    accuracy = accuracy_score(y_test_pd, y_pred)
    f1 = f1_score(y_test_pd, y_pred, average="weighted")
    report = classification_report(y_test_pd, y_pred, output_dict=True)

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Classification Report:\n{json.dumps(report, indent=2)}")

    # Confusion Matrix
    cm = confusion_matrix(y_test_pd, y_pred)
    _plot_confusion_matrix(cm)

    # Feature Importance
    importance = model.feature_importances_
    _plot_feature_importance(importance, feature_names)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "feature_importance": dict(zip(feature_names, importance)),
    }


def _generate_report(metrics: dict[str, Any]) -> None:
    """Generate a markdown report of the evaluation."""
    logging.info("Generating evaluation report...")

    report_content = f"""# Wine Classification Model Evaluation Report

## Model Performance
- **Accuracy**: {metrics['accuracy']:.4f}
- **F1 Score (Weighted)**: {metrics['f1_score']:.4f}

## Classification Report
```json
{json.dumps(metrics['report'], indent=2)}
```

## visualizations
![Confusion Matrix](plots/confusion_matrix.png)

![Feature Importance](plots/feature_importance.png)

## Top Features
"""
    sorted_features = sorted(
        metrics["feature_importance"].items(), key=lambda x: x[1], reverse=True
    )
    for feature, importance in sorted_features[:5]:
        report_content += f"- **{feature}**: {importance:.4f}\n"

    with open(REPORT_FILE, "w") as f:
        f.write(report_content)

    logging.info(f"Report saved to {REPORT_FILE}")


def main() -> None:
    """Main execution flow."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost for Wine Classification"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    _setup_logging(args.debug)

    try:
        X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()

        best_model = _optimize_hyperparameters(X_train, y_train)

        metrics = _evaluate_model(best_model, X_test, y_test, feature_names)

        _generate_report(metrics)

        logging.info("Training pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
