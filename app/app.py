import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing import (
    load_data,
    train_test_split,
    encode_categorical,
    calculate_mean,
    calculate_std,
    standardize,
)

from src.logistic_regression import (
    LogisticRegression,
    predict_probability,
    probability_to_class,
)

from src.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ChurnLab",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("ChurnLab")

st.write("Logistic Regression from scratch for customer churn prediction.")


# ============================================================
# LOAD DATA
# ============================================================


@st.cache_data
def get_data():
    dataset_path = PROJECT_ROOT / "data" / "dataset.csv"
    return load_data(dataset_path)


df = get_data()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Customers", len(df))

with col2:
    st.metric("Features", len(df.columns) - 1)

with col3:
    churn_rate = df["churn"].mean()
    st.metric("Churn Rate", f"{churn_rate:.1%}")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop("churn", axis=1)
y = df["churn"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

X_train_encoded, X_test_encoded = encode_categorical(
    X_train,
    X_test,
)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X_train_num = X_train_encoded.to_numpy(dtype=float)

X_test_num = X_test_encoded.to_numpy(dtype=float)

y_train_num = y_train.to_numpy(dtype=int)

y_test_num = y_test.to_numpy(dtype=int)


# ============================================================
# STANDARDIZATION
# ============================================================

# Mean and standard deviation are calculated
# ONLY from the training data.

train_mean = calculate_mean(X_train_num)

train_std = calculate_std(X_train_num, train_mean)

X_train_scaled = standardize(X_train_num, train_mean, train_std)

X_test_scaled = standardize(X_test_num, train_mean, train_std)


# ============================================================
# MODEL TRAINING
# ============================================================

st.header("Model")

model = LogisticRegression(learning_rate=0.1, epochs=1000)

model.fit(X_train_scaled, y_train_num)


# ============================================================
# TEST PREDICTIONS
# ============================================================

# IMPORTANT:
# predict_probability is a standalone function.

test_probabilities = predict_probability(X_test_scaled, model.weights, model.bias)

# probability_to_class is also a standalone function.

test_predictions = probability_to_class(test_probabilities)


# ============================================================
# METRICS
# ============================================================

model_accuracy = accuracy(y_test_num, test_predictions)

model_precision = precision(y_test_num, test_predictions)

model_recall = recall(y_test_num, test_predictions)

model_f1 = f1_score(y_test_num, test_predictions)


# ============================================================
# DISPLAY METRICS
# ============================================================

st.header("Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{model_accuracy:.1%}")

with col2:
    st.metric("Precision", f"{model_precision:.1%}")

with col3:
    st.metric("Recall", f"{model_recall:.1%}")

with col4:
    st.metric("F1 Score", f"{model_f1:.1%}")


# ============================================================
# TEST RESULTS
# ============================================================

st.header("Test Predictions")

results = pd.DataFrame(
    {
        "Actual": y_test_num,
        "Probability": np.round(test_probabilities, 4),
        "Prediction": test_predictions,
    }
)

st.dataframe(
    results,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# TRAINING LOSS
# ============================================================

st.header("Training Loss")

if hasattr(model, "loss_history"):

    loss_df = pd.DataFrame(
        {
            "Epoch": range(1, len(model.loss_history) + 1),
            "Loss": model.loss_history,
        }
    )

    st.line_chart(
        loss_df,
        x="Epoch",
        y="Loss",
    )


# ============================================================
# LEARNED WEIGHTS
# ============================================================

st.header("Learned Feature Weights")

feature_names = list(X_train_encoded.columns)

weights_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Weight": model.weights,
    }
)

st.bar_chart(
    weights_df,
    x="Feature",
    y="Weight",
)


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

st.header("Predict Customer Churn")

st.write("Enter customer information to estimate churn probability.")


col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1,
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=100,
        value=12,
        step=1,
    )


with col2:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=1000.0,
        value=70.0,
        step=1.0,
    )

    support_calls = st.number_input(
        "Support Calls",
        min_value=0,
        max_value=50,
        value=2,
        step=1,
    )


contract_type = st.selectbox(
    "Contract Type",
    [
        "month-to-month",
        "one-year",
        "two-year",
    ],
)


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

if st.button("Predict Churn", type="primary"):

    customer = pd.DataFrame(
        [
            {
                "age": age,
                "tenure": tenure,
                "monthly_charges": monthly_charges,
                "support_calls": support_calls,
                "contract_type": contract_type,
            }
        ]
    )

    # Encode using the SAME project encoding function.
    #
    # The second argument is required by your
    # encode_categorical(X_train, X_test) function.

    customer_encoded, _ = encode_categorical(customer, customer.copy())

    customer_num = customer_encoded.to_numpy(dtype=float)

    # Use training mean/std.
    customer_scaled = standardize(customer_num, train_mean, train_std)

    # Standalone prediction function.

    probability = predict_probability(customer_scaled, model.weights, model.bias)[0]

    # Standalone class conversion function.

    prediction = probability_to_class(np.array([probability]))[0]

    st.divider()

    if prediction == 1:

        st.error(f"High churn risk — " f"Probability: {probability:.2%}")

    else:

        st.success(f"Low churn risk — " f"Probability: {probability:.2%}")
