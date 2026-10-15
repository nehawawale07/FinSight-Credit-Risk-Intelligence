import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

st.set_page_config(
    page_title="FinSight | Credit Risk Intelligence",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0b1020;
}
[data-testid="stMetric"] {
    background: #151d32;
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "finsight_catboost.cbm")
META_PATH = os.path.join(BASE_DIR, "model_metadata.json")
DEFAULTS_PATH = os.path.join(BASE_DIR, "applicant_defaults.json")

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

st.title("📊 FinSight")
st.subheader("Explainable Credit Risk Intelligence")
st.caption(
    "An educational machine-learning prototype using historical "
    "Home Credit data. Not a lending or financial-advice system."
)

if not all(os.path.exists(p) for p in [
    MODEL_PATH, META_PATH, DEFAULTS_PATH
]):
    st.error(
        "Required model files are missing. Keep app.py, "
        "finsight_catboost.cbm, model_metadata.json and "
        "applicant_defaults.json in the same folder."
    )
    st.stop()

model = load_model()
metadata = load_json(META_PATH)
defaults = load_json(DEFAULTS_PATH)

feature_columns = metadata["feature_columns"]
categorical_features = metadata["categorical_features"]

st.info(
    "This demo fills unedited fields with typical training-data values. "
    "The displayed probability is a model output, not a verified "
    "individual risk assessment. Do not use it to approve or reject loans."
)

st.sidebar.header("Applicant inputs")
st.sidebar.write(
    "Change the available inputs below. Other features retain their "
    "saved reference values."
)

# Start with one value for every feature used during training.
applicant = defaults.copy()

def get_default(name, fallback=0.0):
    value = defaults.get(name, fallback)
    try:
        value = float(value)
        return value if np.isfinite(value) else fallback
    except (TypeError, ValueError):
        return fallback

# Financial fields
if "AMT_INCOME_TOTAL" in feature_columns:
    applicant["AMT_INCOME_TOTAL"] = st.sidebar.number_input(
        "Annual income",
        min_value=0.0,
        value=max(0.0, get_default("AMT_INCOME_TOTAL")),
        step=10000.0,
        help="Annual income in the dataset's currency units."
    )

if "AMT_CREDIT" in feature_columns:
    applicant["AMT_CREDIT"] = st.sidebar.number_input(
        "Requested credit amount",
        min_value=0.0,
        value=max(0.0, get_default("AMT_CREDIT")),
        step=10000.0
    )

if "AMT_ANNUITY" in feature_columns:
    applicant["AMT_ANNUITY"] = st.sidebar.number_input(
        "Annual annuity / payment amount",
        min_value=0.0,
        value=max(0.0, get_default("AMT_ANNUITY")),
        step=1000.0
    )

if "AMT_GOODS_PRICE" in feature_columns:
    applicant["AMT_GOODS_PRICE"] = st.sidebar.number_input(
        "Goods price",
        min_value=0.0,
        value=max(0.0, get_default("AMT_GOODS_PRICE")),
        step=10000.0
    )

if "AGE_YEARS" in feature_columns:
    age = st.sidebar.slider(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=35
    )
    applicant["AGE_YEARS"] = float(age)

    # Keep the original model feature consistent with the age input.
    if "DAYS_BIRTH" in feature_columns:
        applicant["DAYS_BIRTH"] = -float(age) * 365.25

# Update derived ratios when their source values are available.
income = applicant.get("AMT_INCOME_TOTAL", 0)
credit = applicant.get("AMT_CREDIT", 0)
annuity = applicant.get("AMT_ANNUITY", 0)

if "CREDIT_INCOME_RATIO" in feature_columns:
    applicant["CREDIT_INCOME_RATIO"] = (
        credit / income if income > 0 else np.nan
    )

if "ANNUITY_INCOME_RATIO" in feature_columns:
    applicant["ANNUITY_INCOME_RATIO"] = (
        annuity / income if income > 0 else np.nan
    )

st.sidebar.divider()
run_prediction = st.sidebar.button(
    "Analyze applicant",
    type="primary",
    use_container_width=True
)

st.header("Applicant overview")

c1, c2, c3 = st.columns(3)
c1.metric("Annual income", f"{applicant.get('AMT_INCOME_TOTAL', 0):,.0f}")
c2.metric("Credit amount", f"{applicant.get('AMT_CREDIT', 0):,.0f}")
c3.metric("Age", f"{applicant.get('AGE_YEARS', 0):.0f} years")

if run_prediction:
    input_df = pd.DataFrame(
        [{col: applicant.get(col, defaults.get(col)) for col in feature_columns}]
    )

    # Match categorical preprocessing used at training time.
    for col in categorical_features:
        if col in input_df.columns:
            input_df[col] = (
                input_df[col]
                .fillna("__MISSING__")
                .astype(str)
            )

    # Ensure numeric inputs are valid.
    numeric_columns = [
        col for col in feature_columns
        if col not in categorical_features
    ]
    for col in numeric_columns:
        input_df[col] = pd.to_numeric(
            input_df[col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    probability = float(model.predict_proba(input_df)[0, 1])

    
    # SHAP-style explanation for this individual prediction
    
    from catboost import Pool

    shap_pool = Pool(
        data=input_df,
        cat_features=categorical_features
    )

    shap_values = model.get_feature_importance(
         data=shap_pool,
         type="ShapValues"
    )[0]


    contributions = pd.DataFrame({
        "Feature": feature_columns,
        "Contribution": shap_values[:-1]
    })

    # Show the features with the largest absolute contributions
    contributions["AbsoluteContribution"] = (
        contributions["Contribution"].abs()
    )

    top_contributions = (
        contributions
        .sort_values("AbsoluteContribution", ascending=False)
        .head(10)
        .sort_values("Contribution")
    )

    st.subheader("Why did the model make this prediction?")
    st.caption(
        "Positive contributions push the model output toward "
        "TARGET=1; negative contributions push it away. "
        "These are contributions to the raw model output, not "
        "percentage-point changes in probability."
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        top_contributions["Feature"],
        top_contributions["Contribution"]
    )
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("SHAP contribution (raw model output)")
    ax.set_title("Top Individual Prediction Contributors")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


    st.divider()
    st.header("Model output")

    m1, m2 = st.columns(2)
    m1.metric("Predicted probability", f"{probability:.1%}")
    m2.metric("Model validation ROC-AUC",
              f"{metadata['validation_roc_auc']:.3f}")

    st.progress(int(round(probability * 100)))
    st.caption(
        "Probability of the dataset's TARGET=1 outcome, based on the "
        "entered values and defaulted features. This is not a calibrated "
        "loan decision or a guarantee of future repayment."
    )

    st.subheader("Model feature importance")
    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.get_feature_importance()
    }).sort_values("Importance", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        importance["Feature"][::-1],
        importance["Importance"][::-1]
    )
    ax.set_xlabel("Importance")
    ax.set_title("Top features used by the model")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Interpretation")
    st.write(
        "Feature importance summarizes which variables the model used "
        "across its predictions. It does not establish causation and "
        "does not explain why this particular prediction was made."
    )

else:
    st.warning(
        "Adjust the applicant inputs in the sidebar and select "
        "**Analyze applicant** to run the model."
    )

st.divider()
st.subheader("Model performance")
p1, p2 = st.columns(2)
p1.metric("Validation ROC-AUC",
          f"{metadata['validation_roc_auc']:.3f}")
p2.metric("Validation Average Precision",
          f"{metadata['validation_average_precision']:.3f}")

st.caption(
    "Validation metrics come from the held-out dataset. Performance "
    "may differ on new populations. This project is for learning and "
    "portfolio demonstration only; it is not validated for real lending."
)

# Model comparison section
comparison_path = os.path.join(BASE_DIR, "model_comparison.csv")

if os.path.exists(comparison_path):
    st.divider()
    st.header("Model comparison")

    comparison_df = pd.read_csv(comparison_path)

    st.write(
        "Compare the validation performance of the baseline model "
        "and the CatBoost model."
    )

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    chart_df = comparison_df.set_index("Model")[
        ["ROC_AUC", "Average_Precision"]
    ]

    st.bar_chart(chart_df)
else:
    st.info(
        "Upload model_comparison.csv to the repository "
        "to enable model comparison."
    )

