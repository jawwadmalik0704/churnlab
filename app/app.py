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

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
    confusion_matrix,
)

from visualization.plots import (
    plot_loss_curve,
    plot_confusion_matrix,
    plot_feature_weights,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ChurnLab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HEADER
# ============================================================

st.title("ChurnLab")

st.caption("Logistic Regression from scratch for customer churn prediction")

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("ChurnLab")

    st.write(
        "A from-scratch machine learning project "
        "for binary customer churn prediction."
    )

    st.divider()

    st.subheader("ML Pipeline")

    st.write("""
        **1. Data**
        
        Load customer churn data.

        **2. Preprocessing**
        
        Split, encode, and standardize data.

        **3. Training**
        
        Train logistic regression using gradient descent.

        **4. Prediction**
        
        Generate churn probabilities and classes.

        **5. Evaluation**
        
        Accuracy, precision, recall, F1, and confusion matrix.
        """)

    st.divider()

    st.caption("Built with Python, NumPy, Pandas, Matplotlib, " "and Streamlit.")


# ============================================================
# LOAD DATA
# ============================================================


@st.cache_data
def get_data(mtime):

    dataset_path = PROJECT_ROOT / "data" / "dataset.csv"

    return load_data(dataset_path)


dataset_path = PROJECT_ROOT / "data" / "dataset.csv"
mtime = os.path.getmtime(dataset_path) if dataset_path.exists() else 0
df = get_data(mtime)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Customers",
        len(df),
    )


with col2:

    st.metric(
        "Features",
        len(df.columns) - 1,
    )


with col3:

    churn_rate = df["churn"].mean()

    st.metric(
        "Churn Rate",
        f"{churn_rate:.1%}",
    )


with st.expander("View Dataset"):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    "churn",
    axis=1,
)

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

# IMPORTANT:
# Mean and standard deviation are calculated
# ONLY from training data.

train_mean = calculate_mean(X_train_num)

train_std = calculate_std(
    X_train_num,
    train_mean,
)

X_train_scaled = standardize(
    X_train_num,
    train_mean,
    train_std,
)

X_test_scaled = standardize(
    X_test_num,
    train_mean,
    train_std,
)


# ============================================================
# MODEL TRAINING
# ============================================================

st.header("Model Training")

@st.cache_resource
def train_model_pipeline(_X_train, _y_train):
    model = LogisticRegression(
        learning_rate=0.1,
        epochs=1000,
    )
    model.fit(_X_train, _y_train)
    return model

model = train_model_pipeline(
    X_train_scaled,
    y_train_num,
)

st.success("Logistic Regression model trained successfully.")


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_probabilities = predict_probability(
    X_test_scaled,
    model.weights,
    model.bias,
)

test_predictions = probability_to_class(test_probabilities)


# ============================================================
# METRICS
# ============================================================

model_accuracy = accuracy(
    y_test_num,
    test_predictions,
)

model_precision = precision(
    y_test_num,
    test_predictions,
)

model_recall = recall(
    y_test_num,
    test_predictions,
)

model_f1 = f1_score(
    y_test_num,
    test_predictions,
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Accuracy",
        f"{model_accuracy:.1%}",
    )


with col2:

    st.metric(
        "Precision",
        f"{model_precision:.1%}",
    )


with col3:

    st.metric(
        "Recall",
        f"{model_recall:.1%}",
    )


with col4:

    st.metric(
        "F1 Score",
        f"{model_f1:.1%}",
    )


# ============================================================
# EVALUATION NOTE
# ============================================================

st.info(f"""
    **Evaluation note:** The test set contains only
    {len(y_test_num)} customers. Because this is a small
    educational dataset, individual predictions can have
    a large effect on the evaluation metrics.
    """)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.header("Confusion Matrix")

tn, fp, fn, tp = confusion_matrix(
    y_test_num,
    test_predictions,
)

cm_col1, cm_col2 = st.columns([1.4, 1])


with cm_col1:

    confusion_fig = plot_confusion_matrix(
        y_test_num,
        test_predictions,
    )

    st.pyplot(
        confusion_fig,
        clear_figure=True,
    )
    plt.close(confusion_fig)


with cm_col2:

    st.subheader("Prediction Breakdown")

    st.metric(
        "True Negatives",
        int(tn),
    )

    st.metric(
        "False Positives",
        int(fp),
    )

    st.metric(
        "False Negatives",
        int(fn),
    )

    st.metric(
        "True Positives",
        int(tp),
    )


# ============================================================
# TEST PREDICTIONS
# ============================================================

st.header("Test Predictions")

results = pd.DataFrame(
    {
        "Actual": y_test_num,
        "Probability": np.round(
            test_probabilities,
            4,
        ),
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

    loss_fig = plot_loss_curve(
        model.loss_history,
    )

    st.pyplot(
        loss_fig,
        clear_figure=True,
    )
    plt.close(loss_fig)


# ============================================================
# LEARNED FEATURE WEIGHTS
# ============================================================

st.header("Learned Feature Weights")

feature_names = list(X_train_encoded.columns)

weights_fig = plot_feature_weights(
    feature_names,
    model.weights,
)

st.pyplot(
    weights_fig,
    clear_figure=True,
)
plt.close(weights_fig)


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

st.header("Predict Customer Churn")

st.info("💡 **Inference Sandbox:** The inputs below calculate the churn probability for a single, new customer. They do **not** add data to the training/test set or change the global Model Performance metrics shown above.")

st.write("Enter customer information to estimate the probability of churn.")


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

if st.button(
    "Predict Churn",
    type="primary",
):

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

    # Same categorical encoding
    # used during model training.

    customer_encoded, _ = encode_categorical(
        customer,
        customer.copy(),
    )

    customer_num = customer_encoded.to_numpy(dtype=float)

    # IMPORTANT:
    # Use training mean/std.
    # Do NOT calculate new statistics
    # from this single customer.

    customer_scaled = standardize(
        customer_num,
        train_mean,
        train_std,
    )

    probability = predict_probability(
        customer_scaled,
        model.weights,
        model.bias,
    )[0]

    prediction = probability_to_class(np.array([probability]))[0]

    st.divider()

    # ========================================================
    # RESULT
    # ========================================================

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.subheader("Prediction")

        if prediction == 1:

            st.error("High Churn Risk")

        else:

            st.success("Low Churn Risk")

    with result_col2:

        st.subheader("Churn Probability")

        st.metric(
            "Probability",
            f"{probability:.2%}",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ChurnLab • Logistic Regression from scratch • "
    "Educational machine-learning project"
)
