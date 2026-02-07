"""XGBoost model training with hyperparameter tuning for Wine Classification."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from feature_engineering import prepare_features
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: Path = Path("output")
RANDOM_STATE: int = 42
CV_FOLDS: int = 5
N_ITER_SEARCH: int = 20


def _get_param_distributions() -> Dict[str, Any]:
    """Get hyperparameter distributions for RandomizedSearchCV.

    Returns:
        Dictionary of parameter distributions
    """
    param_distributions = {
        "n_estimators": [50, 100, 150, 200, 300],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 3, 4, 5],
        "gamma": [0, 0.1, 0.2, 0.3, 0.4],
        "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0, 2.5],
    }
    return param_distributions


def _perform_cross_validation(
    model: XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, float]:
    """Perform 5-fold cross-validation and log results.

    Args:
        model: XGBoost classifier
        X_train: Training features
        y_train: Training labels

    Returns:
        Dictionary of cross-validation scores
    """
    logging.info(f"Performing {CV_FOLDS}-fold cross-validation...")

    cv_accuracy = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="accuracy")
    cv_precision = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="precision_macro")
    cv_recall = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="recall_macro")
    cv_f1 = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="f1_macro")

    cv_results = {
        "accuracy_mean": float(cv_accuracy.mean()),
        "accuracy_std": float(cv_accuracy.std()),
        "accuracy_scores": cv_accuracy.tolist(),
        "precision_mean": float(cv_precision.mean()),
        "precision_std": float(cv_precision.std()),
        "precision_scores": cv_precision.tolist(),
        "recall_mean": float(cv_recall.mean()),
        "recall_std": float(cv_recall.std()),
        "recall_scores": cv_recall.tolist(),
        "f1_mean": float(cv_f1.mean()),
        "f1_std": float(cv_f1.std()),
        "f1_scores": cv_f1.tolist(),
    }

    logging.info(
        f"Cross-validation Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})"
    )
    logging.info(
        f"Cross-validation Precision: {cv_precision.mean():.4f} (+/- {cv_precision.std():.4f})"
    )
    logging.info(f"Cross-validation Recall: {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")
    logging.info(f"Cross-validation F1-score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")

    for fold_idx in range(CV_FOLDS):
        logging.info(
            f"Fold {fold_idx + 1}: Accuracy={cv_accuracy[fold_idx]:.4f}, "
            f"Precision={cv_precision[fold_idx]:.4f}, "
            f"Recall={cv_recall[fold_idx]:.4f}, "
            f"F1={cv_f1[fold_idx]:.4f}"
        )

    return cv_results


def _tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Perform hyperparameter tuning using RandomizedSearchCV.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Tuple of (best_model, tuning_results)
    """
    logging.info("=" * 80)
    logging.info("HYPERPARAMETER TUNING")
    logging.info("=" * 80)

    base_model = XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )

    param_distributions = _get_param_distributions()

    logging.info(
        f"Performing RandomizedSearchCV with {N_ITER_SEARCH} iterations and {CV_FOLDS}-fold CV"
    )

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH,
        cv=CV_FOLDS,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    random_search.fit(X_train, y_train)

    logging.info(f"Best cross-validation F1-score: {random_search.best_score_:.4f}")
    logging.info(f"Best hyperparameters:\n{json.dumps(random_search.best_params_, indent=2)}")

    tuning_results = {
        "best_score": float(random_search.best_score_),
        "best_params": random_search.best_params_,
        "n_iterations": N_ITER_SEARCH,
        "cv_folds": CV_FOLDS,
        "cv_results": {
            "mean_test_score": random_search.cv_results_["mean_test_score"].tolist(),
            "std_test_score": random_search.cv_results_["std_test_score"].tolist(),
            "params": [str(p) for p in random_search.cv_results_["params"]],
        },
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    tuning_path = OUTPUT_DIR / "tuning_results.json"
    with open(tuning_path, "w") as f:
        json.dump(tuning_results, f, indent=2, default=str)
    logging.info(f"Tuning results saved to {tuning_path}")

    return random_search.best_estimator_, tuning_results


def train_model() -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Train XGBoost classifier with hyperparameter tuning and cross-validation.

    Returns:
        Tuple of (trained_model, training_metrics)
    """
    logging.info("=" * 80)
    logging.info("STARTING MODEL TRAINING")
    logging.info("=" * 80)

    X_train, X_test, y_train, y_test, feature_names = prepare_features()

    logging.info(f"Training set shape: {X_train.shape}")
    logging.info(f"Number of features: {len(feature_names)}")
    logging.info(f"Number of classes: {len(np.unique(y_train))}")

    best_model, tuning_results = _tune_hyperparameters(X_train, y_train)

    logging.info("=" * 80)
    logging.info("CROSS-VALIDATION WITH BEST MODEL")
    logging.info("=" * 80)

    cv_results = _perform_cross_validation(best_model, X_train, y_train)

    logging.info("=" * 80)
    logging.info("TRAINING FINAL MODEL ON FULL TRAINING SET")
    logging.info("=" * 80)

    best_model.fit(X_train, y_train)
    logging.info("Model training complete")

    y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average="macro")
    train_recall = recall_score(y_train, y_train_pred, average="macro")
    train_f1 = f1_score(y_train, y_train_pred, average="macro")

    logging.info("Training set performance:")
    logging.info(f"  Accuracy: {train_accuracy:.4f}")
    logging.info(f"  Precision (macro): {train_precision:.4f}")
    logging.info(f"  Recall (macro): {train_recall:.4f}")
    logging.info(f"  F1-score (macro): {train_f1:.4f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = OUTPUT_DIR / "trained_model.pkl"
    joblib.dump(best_model, model_path)
    logging.info(f"Model saved to {model_path}")

    training_metrics = {
        "cv_results": cv_results,
        "tuning_results": tuning_results,
        "train_accuracy": float(train_accuracy),
        "train_precision": float(train_precision),
        "train_recall": float(train_recall),
        "train_f1": float(train_f1),
        "model_path": str(model_path),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_classes": int(len(np.unique(y_train))),
    }

    logging.info("=" * 80)
    logging.info("MODEL TRAINING COMPLETE")
    logging.info("=" * 80)

    return best_model, training_metrics


if __name__ == "__main__":
    model, metrics = train_model()
    logging.info("Training pipeline executed successfully")
