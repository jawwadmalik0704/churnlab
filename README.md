# ChurnLab

## Logistic Regression From Scratch for Customer Churn Prediction

**ChurnLab** is a machine learning learning project focused on understanding the complete workflow behind **Logistic Regression and binary classification**.

The core model, preprocessing pipeline, evaluation metrics, and training workflow were implemented manually to understand what happens inside a machine learning system instead of treating the model as a black box.

### Live Demo

**[Launch ChurnLab](https://churnlab-eppwvmeo5shhmntfhg5thl.streamlit.app/)**

---

## Project Goal

The main purpose of ChurnLab was **learning machine learning fundamentals through implementation**.

The Streamlit application is only the final interface. The main learning objective was understanding the relationship between:

```text
Raw Data
    ↓
Data Analysis & Preprocessing
    ↓
Mathematics
    ↓
Machine Learning Algorithm
    ↓
Model Training
    ↓
Prediction
    ↓
Evaluation
    ↓
Visualization
    ↓
Interactive Application
```

Instead of simply using a pre-built Logistic Regression implementation, the project explores the important components behind the algorithm.

---

# What Was Implemented From Scratch?

The project manually implements the major parts of a Logistic Regression workflow, including:

- Model weights
- Model bias
- Sigmoid function
- Forward prediction
- Probability calculation
- Loss calculation
- Gradient calculation
- Gradient descent
- Weight updates
- Bias updates
- Training over multiple epochs
- Probability-to-class conversion
- Classification metrics
- Confusion matrix

The project does **not** use `sklearn.linear_model.LogisticRegression` for the core model training.

The purpose is to understand the mathematics and algorithmic process behind Logistic Regression.

---

# Complete Machine Learning Pipeline

The complete workflow is:

```text
Customer Dataset
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
Linear Prediction
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
Visualizations
      ↓
Streamlit Application
```

---

# Dataset

The project uses a small educational customer churn dataset.

The dataset contains features such as:

- Age
- Tenure
- Monthly Charges
- Support Calls
- Contract Type
- Churn

The target variable is:

```text
churn
```

where:

```text
0 = Customer did not churn
1 = Customer churned
```

The dataset is intentionally small because the primary objective is understanding the machine learning workflow rather than achieving production-level predictive performance.

---

# Data Preprocessing

The preprocessing pipeline includes several important steps.

## 1. Data Loading

The dataset is loaded from:

```text
data/dataset.csv
```

using Pandas.

## 2. Train/Test Split

The dataset is divided into training and testing subsets.

The model learns from the training data and is then evaluated using previously unseen test samples.

## 3. Categorical Encoding

The `contract_type` feature contains:

```text
month-to-month
one-year
two-year
```

These categorical values are converted into numerical indicator features so that they can be used by the mathematical model.

## 4. Feature Standardization

The numerical features are standardized using statistics calculated from the training data.

The training mean and standard deviation are then reused when transforming:

- Training data
- Test data
- New customer input

This keeps the preprocessing process consistent.

---

# Logistic Regression

The project implements Logistic Regression as a binary classification algorithm.

The model learns:

```text
Weights
+
Bias
```

and uses the input features to calculate a linear score.

That score is transformed into a probability using the sigmoid function.

Conceptually:

```text
Features
   ↓
Weighted Combination
   ↓
Sigmoid
   ↓
Probability
   ↓
Class Prediction
```

The resulting probability represents the model's estimated likelihood that the customer belongs to the churn class.

---

# Model Training

The model is trained using gradient descent.

The training process repeatedly performs:

```text
1. Calculate prediction
        ↓
2. Calculate loss
        ↓
3. Calculate gradients
        ↓
4. Update weights
        ↓
5. Update bias
        ↓
6. Repeat for next epoch
```

The training loss is stored during training so that the optimization process can be visualized.

---

# Model Evaluation

The project implements several classification metrics.

## Accuracy

Measures the proportion of total predictions that are correct.

## Precision

Measures how many customers predicted as churners were actually churners.

## Recall

Measures how many actual churners were successfully detected.

## F1 Score

Combines precision and recall into a single metric.

## Confusion Matrix

The confusion matrix separates predictions into:

```text
                 Predicted
                 0       1

Actual 0        TN      FP
Actual 1        FN      TP
```

This gives a more detailed view of the model's classification behavior than accuracy alone.

---

# Visualizations

The project includes visualizations for understanding model behavior.

### Training Loss

Shows how the loss changes over training epochs.

### Confusion Matrix

Shows the relationship between actual and predicted classes.

### Feature Weights

Shows the learned weight associated with each input feature.

These visualizations help connect the mathematical training process with observable model behavior.

---

# Streamlit Application

The final machine learning pipeline is connected to an interactive Streamlit application.

The application provides:

- Dataset overview
- Customer count
- Feature count
- Churn rate
- Model performance metrics
- Test predictions
- Prediction probabilities
- Training loss visualization
- Learned feature weights
- Interactive customer prediction

The user can enter:

- Age
- Tenure
- Monthly Charges
- Support Calls
- Contract Type

and receive:

```text
Churn Prediction
+
Churn Probability
```

---

# Live Application

### Try ChurnLab

**[Open the deployed application](https://churnlab-eppwvmeo5shhmntfhg5thl.streamlit.app/)**

The deployed application runs from the project's `main` branch using Streamlit Community Cloud.

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
- Prediction
- Loss calculation
- Gradient calculation
- Gradient descent
- Weight updates
- Bias updates
- Training history

---

## `src/preprocessing.py`

Contains the preprocessing pipeline.

Responsible for:

- Loading data
- Train/test splitting
- Categorical encoding
- Mean calculation
- Standard deviation calculation
- Feature standardization

---

## `src/metrics.py`

Contains the model evaluation functions:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## `src/train.py`

Contains the training workflow that connects preprocessing and model training.

It demonstrates how the individual components of the project connect together to train the Logistic Regression model.

---

## `visualization/plots.py`

Contains reusable visualization functions for:

- Training loss
- Confusion matrix
- Feature weights

---

## `notebooks/experiments.ipynb`

Contains the learning and experimentation workflow.

The notebook demonstrates the mathematical and machine learning concepts used throughout the project.

---

## `app/app.py`

Contains the Streamlit application.

It connects the project components into an interactive interface:

```text
Dataset
   ↓
Preprocessing
   ↓
Model
   ↓
Prediction
   ↓
Evaluation
   ↓
Visualization
   ↓
User Interface
```

---

# Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Jupyter Notebook
- Git
- GitHub

The core Logistic Regression model and evaluation logic were implemented manually for educational purposes.

---

# Running the Project Locally

## 1. Clone the repository

```bash
git clone https://github.com/jawwadmalik0704/churnlab.git
```

## 2. Enter the project

```bash
cd churnlab
```

## 3. Create a virtual environment

### Windows

```powershell
python -m venv .venv
```

## 4. Activate the environment

```powershell
.venv\Scripts\activate
```

## 5. Install dependencies

```powershell
pip install -r requirements.txt
```

## 6. Run the Streamlit application

Run the command from the **project root**:

```powershell
streamlit run app/app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# How to Use the Application

1. Open the Streamlit application.
2. Review the dataset overview.
3. Review the model performance metrics.
4. Inspect the training-loss graph.
5. Inspect the learned feature weights.
6. Review the test predictions.
7. Go to **Predict Customer Churn**.
8. Enter customer information.
9. Click **Predict Churn**.
10. Review the predicted risk and probability.

---

# Five-Day Learning Journey

ChurnLab was developed as a five-day project-based learning progression.

## Day 1 — Data Foundation

Learned and implemented:

- Dataset structure
- Features and target
- CSV data loading
- Customer churn data
- Basic data handling

## Day 2 — Preprocessing & Sigmoid

Learned and implemented:

- Train/test splitting
- Categorical encoding
- Numerical conversion
- Feature standardization
- Mean and standard deviation
- Sigmoid function

## Day 3 — Model Training

Learned and implemented:

- Logistic Regression
- Weights
- Bias
- Forward prediction
- Loss
- Gradients
- Gradient descent
- Parameter updates
- Training epochs

## Day 4 — Evaluation & Visualization

Learned and implemented:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion matrix
- Training-loss visualization
- Feature-weight visualization

## Day 5 — Final Application

Focused on:

- Connecting all components
- Building the Streamlit interface
- Interactive customer prediction
- Displaying model results
- Integrating visualizations
- Testing the complete pipeline
- Git/GitHub integration
- Final release
- Deployment

---

# What I Learned

The most important outcome of ChurnLab was understanding that a machine learning model is not simply:

```python
model.fit(X, y)
```

Instead, a complete machine learning workflow involves multiple connected stages.

### Data

I learned how raw structured data must be prepared before a model can use it.

### Mathematics

I worked with concepts including:

- Linear combinations
- Mean
- Standard deviation
- Standardization
- Sigmoid
- Probability
- Loss
- Gradients
- Optimization

### Machine Learning

I learned how:

- Logistic Regression performs binary classification
- Model parameters are learned
- Gradient descent optimizes parameters
- Probabilities become class predictions
- Classification metrics evaluate predictions

### Data Science

I practiced:

- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
- Data preprocessing
- Visualization
- Model evaluation

### Machine Learning Engineering

I learned how to organize a machine learning project into separate modules and connect:

```text
Data
+
Preprocessing
+
Model
+
Training
+
Evaluation
+
Visualization
+
Application
```

### Software Engineering

The project also provided practical experience with:

- Git
- GitHub
- Branches
- Feature branches
- Pull requests
- Merging
- Dependency management
- Project structure
- Streamlit deployment

---

# Important Learning Principle

ChurnLab was intentionally designed to avoid treating machine learning as a black box.

The goal was to understand the connection:

```text
Mathematics
      ↓
Algorithm
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

This makes ChurnLab primarily a **machine learning learning project**, with the Streamlit application serving as the final demonstration layer.

---

# Limitations

ChurnLab is an educational project and is not intended to be a production-grade churn prediction system.

The dataset is intentionally small and simplified.

Therefore:

- The dataset is not representative of a real telecommunications population.
- Model performance should not be interpreted as real-world performance.
- A production system would require substantially more data.
- Additional validation would be required.
- Feature engineering would need to be more comprehensive.
- Model comparison would be necessary.
- Monitoring and retraining would be required for a real deployment.

The primary purpose is understanding the machine learning workflow.

---

# Future Improvements

Possible future improvements include:

- Larger real-world datasets
- Cross-validation
- Hyperparameter tuning
- Class-imbalance handling
- More feature engineering
- Additional classification algorithms
- Model comparison
- Automated testing
- Improved monitoring
- Production-grade data pipelines

---

# Repository

**GitHub:**
https://github.com/jawwadmalik0704/churnlab

**Live Application:**
https://churnlab-eppwvmeo5shhmntfhg5thl.streamlit.app/

---

# Author

**Mohammed Jabal Malik**

Machine Learning / AI Engineering Learning Project

---

# License

This project is primarily intended for educational and portfolio purposes.
