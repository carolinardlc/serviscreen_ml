# Cervical Cancer Detection with Machine Learning

Academic classification project on the **UCI Cervical Cancer (Risk Factors)** dataset.
The goal is to estimate the risk of a **positive biopsy** result from demographic, lifestyle,
and clinical-history risk factors, as a **screening support** tool.

## Repository contents

| File | Description |
|---|---|
| `proyecto_cervical_cancer.ipynb` | Main notebook, runnable end to end. |
| `risk_factors_cervical_cancer.csv` | Dataset (858 rows × 36 columns). |
| `requirements.txt` | Project dependencies. |
| `README.md` | This file. |

## Problem characteristics

- **Target:** `Biopsy` (standard in the literature).
- **Data leakage avoided:** `Hinselmann`, `Schiller`, and `Citology` are other diagnostic
  tests and are **NOT** used as *features*.
- **Strong imbalance:** only **~6.4%** positive cases ⇒ evaluation prioritizes
  **Recall** and **PR-AUC**, not *accuracy*.
- **Missing values:** encoded as `"?"` → converted to `NaN`; two columns with ~92% nulls are
  dropped.

## Notebook structure

- **Setup** — dependencies and imports
- **Data Loading and Cleaning** — `?`→NaN and numeric conversion
- **Data Analysis** — missing values, distributions, outliers, imbalance, correlations, and removal of *leakage* columns
- **Data Transformation** — imputation, scaling, stratified split, and SMOTE without *leakage*
- **Feature Selection** — `SelectKBest`, RF importances, and **genetic algorithm (DEAP)**
- **Models Used** — Logistic Regression, Random Forest, XGBoost, LightGBM
- **Tests with All Features** — tuning (`RandomizedSearchCV` + `StratifiedKFold`, PR-AUC scoring), comparative evaluation, and the GA-features variant
- **Interpretability (SHAP)**
- **Conclusions** — findings, limitations, and improvements

## How to run it

```bash
# 1. Create the environment (Python 3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate

# 2. (macOS only) OpenMP runtime for xgboost/lightgbm
brew install libomp

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook proyecto_cervical_cancer.ipynb
```

## Reproducibility

All seeds are fixed (`RANDOM_STATE = 42`). Preprocessing (imputation, scaling, and
SMOTE) is performed **inside pipelines** to avoid *data leakage* during cross-validation.
