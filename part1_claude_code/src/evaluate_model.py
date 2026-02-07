"""Model evaluation and performance analysis for Wine Classification."""

import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from feature_engineering import prepare_features
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: Path = Path("output")
CLASS_NAMES: list[str] = ["Class 0", "Class 1", "Class 2"]


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Plot confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Confusion matrix plot saved to {output_path}")


def _plot_roc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
) -> None:
    """Plot ROC curves for multiclass classification (one-vs-rest).

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for each class
        output_path: Path to save the plot
    """
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y_true_bin.shape[1]

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"ROC curves plot saved to {output_path}")


def _plot_precision_recall_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
) -> None:
    """Plot precision-recall curves for multiclass classification.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for each class
        output_path: Path to save the plot
    """
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y_true_bin.shape[1]

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{CLASS_NAMES[i]}",
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Precision-recall curves plot saved to {output_path}")


def _plot_feature_importance(
    model,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Plot feature importance from XGBoost model.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    importance_scores = model.feature_importances_
    indices = np.argsort(importance_scores)[::-1]

    top_n = min(20, len(feature_names))
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = importance_scores[top_indices]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_scores[::-1], color="#1f77b4", alpha=0.8)
    plt.yticks(range(top_n), top_features[::-1])
    plt.xlabel("Feature Importance", fontsize=12)
    plt.title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Feature importance plot saved to {output_path}")


def evaluate_model() -> Dict:
    """Evaluate trained XGBoost model and generate comprehensive metrics.

    Returns:
        Dictionary containing all evaluation metrics
    """
    logging.info("=" * 80)
    logging.info("STARTING MODEL EVALUATION")
    logging.info("=" * 80)

    model_path = OUTPUT_DIR / "trained_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    model = joblib.load(model_path)
    logging.info(f"Loaded model from {model_path}")

    X_train, X_test, y_train, y_test, feature_names = prepare_features()
    logging.info(f"Test set shape: {X_test.shape}")

    logging.info("=" * 80)
    logging.info("GENERATING PREDICTIONS")
    logging.info("=" * 80)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    logging.info("=" * 80)
    logging.info("CALCULATING METRICS")
    logging.info("=" * 80)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro")
    precision_weighted = precision_score(y_test, y_pred, average="weighted")
    recall_macro = recall_score(y_test, y_pred, average="macro")
    recall_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.info(f"Precision (macro): {precision_macro:.4f}")
    logging.info(f"Precision (weighted): {precision_weighted:.4f}")
    logging.info(f"Recall (macro): {recall_macro:.4f}")
    logging.info(f"Recall (weighted): {recall_weighted:.4f}")
    logging.info(f"F1-score (macro): {f1_macro:.4f}")
    logging.info(f"F1-score (weighted): {f1_weighted:.4f}")

    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)

    logging.info("=" * 80)
    logging.info("PER-CLASS METRICS")
    logging.info("=" * 80)

    for i, class_name in enumerate(CLASS_NAMES):
        logging.info(f"{class_name}:")
        logging.info(f"  Precision: {precision_per_class[i]:.4f}")
        logging.info(f"  Recall: {recall_per_class[i]:.4f}")
        logging.info(f"  F1-score: {f1_per_class[i]:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    logging.info("=" * 80)
    logging.info("CONFUSION MATRIX")
    logging.info("=" * 80)
    logging.info(f"\n{cm}")

    y_true_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc_per_class = []
    for i in range(len(CLASS_NAMES)):
        roc_auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc_per_class.append(float(roc_auc))
        logging.info(f"ROC AUC for {CLASS_NAMES[i]}: {roc_auc:.4f}")

    classification_rep = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    logging.info("=" * 80)
    logging.info("CLASSIFICATION REPORT")
    logging.info("=" * 80)
    logging.info(f"\n{classification_rep}")

    logging.info("=" * 80)
    logging.info("GENERATING VISUALIZATIONS")
    logging.info("=" * 80)

    OUTPUT_DIR.mkdir(exist_ok=True)

    _plot_confusion_matrix(y_test, y_pred, OUTPUT_DIR / "confusion_matrix.png")
    _plot_roc_curves(y_test, y_pred_proba, OUTPUT_DIR / "roc_curves.png")
    _plot_precision_recall_curves(y_test, y_pred_proba, OUTPUT_DIR / "precision_recall_curves.png")
    _plot_feature_importance(model, feature_names, OUTPUT_DIR / "feature_importance.png")

    evaluation_metrics = {
        "test_accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "precision_weighted": float(precision_weighted),
        "recall_macro": float(recall_macro),
        "recall_weighted": float(recall_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "roc_auc_per_class": roc_auc_per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_rep,
        "n_test_samples": int(len(y_test)),
    }

    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(evaluation_metrics, f, indent=2, default=str)
    logging.info(f"Evaluation metrics saved to {metrics_path}")

    logging.info("=" * 80)
    logging.info("MODEL EVALUATION COMPLETE")
    logging.info("=" * 80)

    return evaluation_metrics


if __name__ == "__main__":
    metrics = evaluate_model()
    logging.info("Evaluation pipeline executed successfully")
