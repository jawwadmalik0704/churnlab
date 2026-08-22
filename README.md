# ChurnLab

### Logistic Regression From Scratch for Customer Churn Prediction

ChurnLab is a five-day machine learning learning project built to understand **what happens inside a machine learning classification pipeline**, rather than simply calling a pre-built machine learning model.

The project implements Logistic Regression from scratch and connects the model to an interactive Streamlit application for customer churn prediction.

> **The main goal of ChurnLab is learning the machine learning process behind the model — the Streamlit application is the interface used to demonstrate that process.**

---

## Live Demo

**Coming soon — deployment in progress.**

---

## Project Overview

ChurnLab takes customer information, processes the data, trains a Logistic Regression model, evaluates its predictions, and exposes the final model through an interactive web application.

The complete pipeline is:

```text
Raw Customer Dataset
        ↓
Data Loading
        ↓
Train / Test Split
        ↓
Categorical Encoding
        ↓
Numerical Conversion
        ↓
Feature Standardization
        ↓
Logistic Regression
        ↓
Sigmoid Probability
        ↓
Loss Calculation
        ↓
Gradient Descent
        ↓
Weight & Bias Updates
        ↓
Model Predictions
        ↓
Evaluation Metrics
        ↓
Visualization
        ↓
Streamlit Application
```

---

## Why This Project?

The purpose of this project was not simply to use a machine learning library and obtain predictions.

The objective was to understand the internal workflow of a classification model:

- How raw data becomes numerical model input
- Why categorical variables need encoding
- Why features are standardized
- How Logistic Regression produces probabilities
- How the sigmoid function is used
- How loss measures model performance
- How gradients are calculated
- How gradient descent updates parameters
- How weights and bias are learned
- How probabilities become class predictions
- How classification metrics evaluate a model
- How a trained model can be connected to an application

The project therefore focuses on **machine learning fundamentals and implementation**, not only the final prediction interface.

---

# Machine Learning Implementation

## Logistic Regression From Scratch

The core Logistic Regression model was implemented manually rather than using a ready-made Logistic Regression estimator from a machine learning library.

The implementation includes:

- Model weights
- Model bias
- Sigmoid function
- Forward prediction
- Loss calculation
- Gradient calculation
- Gradient descent
- Parameter updates
- Training over multiple epochs
- Probability prediction
- Conversion of probabilities into binary classes

The model learns parameters from the training data and uses them to predict whether a customer is likely to churn.

---

## Training Process

During training, the model repeatedly performs the following process:

```text
Input Features
      ↓
Linear Combination
      ↓
Sigmoid Function
      ↓
Predicted Probability
      ↓
Loss
      ↓
Gradients
      ↓
Update Weights & Bias
      ↓
Next Epoch
```

The training process records the loss over the epochs so that the optimization process can be visualized.

---

# Data Processing

The project includes a preprocessing pipeline implemented without relying on a pre-built preprocessing pipeline.

### 1. Data Loading

The customer dataset is loaded from:

```text
data/dataset.csv
```

### 2. Train/Test Split

The data is divided into training and testing sets.

The test data is kept separate from the training process so that the model can be evaluated on unseen samples.

### 3. Categorical Encoding

The `contract_type` feature contains categorical values:

```text
month-to-month
one-year
two-year
```

These categories are converted into numerical indicator features.

### 4. Feature Standardization

Numerical features are standardized using the training-set mean and standard deviation.

The training statistics are then reused when transforming the test data and new customer inputs.

This prevents the test/customer data from influencing the training preprocessing statistics.

---

# Model Evaluation

The project implements classification evaluation metrics to measure model performance.

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The confusion matrix separates predictions into:

```text
                 Predicted
                 0       1

Actual 0        TN      FP
Actual 1        FN      TP
```

These metrics provide different perspectives on classification performance instead of relying only on accuracy.

---

# Visualizations

The project includes visualizations for understanding the model and its training process.

### Training Loss

Shows how the training loss changes across epochs.

### Confusion Matrix

Shows the relationship between actual and predicted classes.

### Feature Weights

Shows the learned Logistic Regression weights for the input features.

These visualizations help connect the mathematical model with observable training and prediction behavior.

---

# Interactive Streamlit Application

The final model is connected to a Streamlit application.

The application provides:

- Dataset overview
- Customer count
- Feature information
- Churn rate
- Model performance metrics
- Test predictions
- Prediction probabilities
- Training-loss visualization
- Learned feature weights
- Interactive customer churn prediction

A user can enter customer information such as:

- Age
- Tenure
- Monthly charges
- Support calls
- Contract type

and receive:

```text
Churn Prediction
+
Churn Probability
```

The application uses the same preprocessing logic and trained model implemented in the project.

---

# Project Structure

```text
churnlab/
│
├── app/
│   └── app.py
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── experiments.ipynb
│
├── src/
│   ├── logistic_regression.py
│   ├── metrics.py
│   ├── preprocessing.py
│   └── train.py
│
├── visualization/
│   └── plots.py
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# File Responsibilities

## `src/logistic_regression.py`

Contains the core Logistic Regression implementation.

Responsible for:

- Model initialization
- Sigmoid calculation
- Forward prediction
- Loss calculation
- Gradient calculation
- Gradient descent
- Weight updates
- Bias updates
- Training history

---

## `src/preprocessing.py`

Contains the data preparation pipeline.

Responsible for:

- Loading the dataset
- Train/test splitting
- Categorical encoding
- Mean calculation
- Standard deviation calculation
- Feature standardization

---

## `src/metrics.py`

Contains the evaluation functions.

Responsible for:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion matrix

---

## `src/train.py`

Contains the training workflow that connects preprocessing and model training.

It demonstrates how the different components of the machine learning pipeline work together.

---

## `visualization/plots.py`

Contains reusable visualization functions for:

- Training loss
- Confusion matrix
- Feature weights

---

## `notebooks/experiments.ipynb`

Contains the learning and experimentation workflow.

The notebook demonstrates the project step by step and provides a practical environment for understanding the model and its results.

---

## `app/app.py`

Contains the Streamlit application.

It connects:

```text
Dataset
   ↓
Preprocessing
   ↓
Model
   ↓
Prediction
   ↓
Metrics
   ↓
Visualizations
   ↓
Interactive UI
```

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Jupyter Notebook
- Git
- GitHub

The core machine learning algorithm was implemented manually to focus on understanding the underlying concepts.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/jawwadmalik0704/churnlab.git
```

Move into the project directory:

```bash
cd churnlab
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

---

# Run the Application Locally

From the project root:

```powershell
streamlit run app/app.py
```

Streamlit will provide a local address similar to:

```text
http://localhost:8501
```

Open that address in your browser.

---

# Using the Application

1. Open the Streamlit application.
2. Review the dataset overview.
3. Review the model performance metrics.
4. Inspect the training-loss visualization.
5. Inspect the learned feature weights.
6. Enter customer information under **Predict Customer Churn**.
7. Click **Predict Churn**.
8. Review the predicted churn risk and probability.

---

# Example Prediction

Example customer:

```text
Age: 35
Tenure: 12
Monthly Charges: 70
Support Calls: 2
Contract Type: month-to-month
```

The model can produce a churn probability and classify the customer as either:

```text
High Churn Risk
```

or:

```text
Low Churn Risk
```

The exact prediction depends on the trained model and preprocessing pipeline.

---

# What I Learned

This project was primarily a learning exercise in machine learning engineering.

### Data Processing

- Loading structured datasets
- Separating features and targets
- Train/test splitting
- Categorical encoding
- Numerical conversion
- Feature standardization

### Mathematics

- Linear combinations
- Sigmoid function
- Probability interpretation
- Loss functions
- Gradients
- Gradient descent
- Parameter optimization

### Machine Learning

- Binary classification
- Logistic Regression
- Model training
- Prediction probabilities
- Classification thresholds
- Evaluation metrics
- Confusion matrices

### Python / Data Science

- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
- Modular Python files
- Connecting multiple modules into one pipeline

### Machine Learning Application Development

- Connecting a trained model to a user interface
- Building an interactive prediction workflow
- Displaying model metrics
- Displaying visualizations
- Handling user input
- Turning an ML experiment into a usable application

### Software Development

- Git
- GitHub
- Branch-based development
- Feature branches
- Pull requests
- Merging development work
- Managing project dependencies
- Organizing a machine learning repository

---

# Five-Day Learning Journey

The project was developed as a five-day progression.

### Day 1 — Data Foundation

Focused on:

- Dataset structure
- Features and target
- Loading data
- Understanding customer churn data

### Day 2 — Preprocessing

Focused on:

- Train/test splitting
- Categorical encoding
- Numerical conversion
- Standardization
- Sigmoid fundamentals

### Day 3 — Model Training

Focused on:

- Logistic Regression implementation
- Weights
- Bias
- Predictions
- Loss
- Gradient descent
- Model training

### Day 4 — Evaluation & Visualization

Focused on:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion matrix
- Training-loss visualization
- Feature-weight visualization

### Day 5 — Final Application

Focused on:

- Connecting all components
- Building the Streamlit interface
- Interactive customer prediction
- Model results visualization
- Final project integration
- Testing
- Git/GitHub release workflow

---

# Important Learning Principle

ChurnLab intentionally avoids treating machine learning as a black box.

Instead of only writing:

```python
model.fit(X, y)
```

the project explores what happens inside the training process.

The goal is to understand the relationship between:

```text
Mathematics
   ↓
Algorithms
   ↓
Python Implementation
   ↓
Data Processing
   ↓
Model Training
   ↓
Evaluation
   ↓
Application
```

This makes ChurnLab a **machine-learning learning project first and a prediction application second**.

---

# Limitations

This project is intentionally small and educational.

The included dataset contains a limited number of customer records and is designed for learning the machine learning workflow rather than producing a production-grade churn prediction system.

Therefore:

- Model performance should not be interpreted as production performance.
- The dataset should not be treated as representative of a real telecommunications customer population.
- Larger and more realistic datasets would be required for a production system.
- Additional validation and model comparison would be required for real-world deployment.

The purpose of the project is to understand the machine learning pipeline and implementation details.

---

# Future Improvements

Possible future improvements include:

- Larger real-world datasets
- Cross-validation
- Hyperparameter tuning
- Multiple classification models
- Better class-imbalance handling
- More extensive feature engineering
- Improved model comparison
- More comprehensive automated testing
- Production-grade deployment

---

# Project Goal

The ultimate goal of ChurnLab was not simply to predict whether a customer will churn.

It was to understand how a machine learning model moves from:

```text
Raw Data
    ↓
Mathematics
    ↓
Algorithm
    ↓
Implementation
    ↓
Training
    ↓
Evaluation
    ↓
Application
```

and to build that complete workflow with a clear understanding of each major component.

---

## Author

**Mohammed Jabal Malik**

Machine Learning / AI Engineering Learning Project

GitHub: [jawwadmalik0704](https://github.com/jawwadmalik0704)

---

## License

This project is intended primarily as an educational and portfolio project.
