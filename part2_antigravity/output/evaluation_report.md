# Wine Classification Model Evaluation Report

## Model Performance
- **Accuracy**: 1.0000
- **F1 Score (Weighted)**: 1.0000

## Classification Report
```json
{
  "0": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 12.0
  },
  "1": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 14.0
  },
  "2": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 10.0
  },
  "accuracy": 1.0,
  "macro avg": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 36.0
  },
  "weighted avg": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 36.0
  }
}
```

## visualizations
![Confusion Matrix](plots/confusion_matrix.png)

![Feature Importance](plots/feature_importance.png)

## Top Features
- **od280/od315_of_diluted_wines**: 0.1761
- **proline**: 0.1553
- **color_intensity**: 0.1418
- **flavanoids**: 0.1108
- **alcohol**: 0.0687
