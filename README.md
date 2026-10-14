# FinSight — Explainable Credit Risk Intelligence

An end-to-end machine learning project that explores credit repayment difficulty using the Home Credit Default Risk dataset. FinSight combines predictive modeling, model evaluation, feature importance, and an interactive Streamlit dashboard.

## Live Demo

**Streamlit App:**  https://finsight-credit-risk-intelligence-fqdtmrpvvapdkxv2agxdib.streamlit.app/

## Project Overview

Credit risk modeling uses historical applicant information to estimate the likelihood of repayment difficulties. FinSight demonstrates a machine learning workflow for exploring this problem, with an emphasis on model evaluation and explainability.

This project is an educational prototype, not a production lending system.

## Features

- Exploratory data analysis and visualization
- Financial feature engineering, including credit-to-income and annuity-to-income ratios
- Logistic Regression baseline
- CatBoost classification model
- Imbalanced-class evaluation using ROC-AUC and Average Precision
- Global feature importance
- Individual prediction contributions using CatBoost SHAP values
- Interactive Streamlit dashboard
- Saved model artifacts for deployment

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- CatBoost
- Streamlit
- GitHub

## Dataset

**Home Credit Default Risk — Kaggle**

https://www.kaggle.com/competitions/home-credit-default-risk/data

The target variable, `TARGET`, represents whether repayment difficulties were recorded in the historical data.

## Model Evaluation

Validation results from the current implementation:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.763 |
| Average Precision | 0.254 |

These are the reported validation metrics for the current model. Results may vary with different data splits, preprocessing, or model configurations. They do not establish performance on future applicants.

## How It Works

1. Load and inspect the historical dataset.
2. Explore missing values and relationships between features.
3. Engineer financial ratios and age-related features.
4. Train and validate classification models.
5. Compare ROC-AUC and Average Precision.
6. Inspect global feature importance and individual SHAP contributions.
7. Load the trained model into the Streamlit dashboard.

## Run Locally

Clone this repository and install the dependencies:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd FinSight-Credit-Risk-Intelligence
pip install -r requirements.txt
streamlit run app.py
```

Ensure the trained CatBoost model and JSON metadata/default files are present in the same directory as `app.py`.

## Limitations and Responsible Use

- The project uses historical data and may reflect biases in that data.
- The dashboard fills unedited features with saved reference values, so an individual prediction is illustrative rather than a complete applicant assessment.
- Feature importance and SHAP contributions explain model behavior; they do not establish causation.
- Validation metrics do not guarantee performance on new populations.
- The application must not be used to make real loan approval or rejection decisions.

## Author

**Neha Wawale**

AI & Data Science | Machine Learning | Python | Data Analytics
