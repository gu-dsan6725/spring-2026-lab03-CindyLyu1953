# ML Pipeline Plan: Wine Classification

## Goal
Build a complete machine learning pipeline to classify wines into 3 classes using the UCI Wine dataset and XGBoost, adhering to strict coding standards.

## Project Structure
```
part2_antigravity/
├── plan.md                 # This file
├── pyproject.toml          # Project dependencies (uv)
├── src/                    # Source code
│   ├── eda.py              # Exploratory Data Analysis
│   ├── features.py         # Feature Engineering
│   └── train.py            # Model Training & Evaluation
└── output/                 # Artifacts (Plots, Reports)
```

## Steps

### 1. Exploratory Data Analysis (EDA)
**Script:** `src/eda.py`
- Load data using `sklearn.datasets.load_wine()`.
- Convert to `polars.DataFrame`.
- **Outputs:**
    - Summary statistics table.
    - Distribution plots (histograms/boxplots) for key features.
    - Correlation heatmap.
    - Class balance check (bar plot).
    - Outlier detection report.

### 2. Feature Engineering
**Script:** `src/features.py` (or integrated into pipeline)
- Create at least 3 derived features (e.g., `proline_magnesium_ratio`, `alcohol_ash_ratio`).
- Apply Standard Scaling to numerical features.
- Perform Stratified Train/Test Split (80/20).

### 3. Model Training
**Script:** `src/train.py`
- **Algorithm**: XGBoost Classifier.
- **Validation**: 5-Fold Cross-Validation.
- **CLI Args:** `--debug` to enable DEBUG level logging.
- **Tuning:** `RandomizedSearchCV` (n_iter=20) for XGBoost hyperparameters.
- **Metrics:** Accuracy, Per-class Precision/Recall, F1-Score.
- **Visualization:** Confusion Matrix Heatmap.

### 4. Evaluation & Reporting
**Output:** `output/evaluation_report.md`
- Compile all metrics and plots into a comprehensive report.
- Analyze feature importance.
- Provide recommendations based on model performance.

## Coding Standards
- **Python**: 3.11+
- **Manager**: `uv`
- **Data**: `polars`
- **Linting**: `ruff`
- **Logging**: Specific format with timestamp and process ID.
- **Type Hints**: Required for all functions.
