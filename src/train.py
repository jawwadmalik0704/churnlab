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

from src.metrics import accuracy, precision, recall, f1_score


def train_model(dataset_path):

    # Load dataset
    df = load_data(dataset_path)

    # Separate features and target
    X = df.drop("churn", axis=1)
    y = df["churn"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Encode categorical features
    X_train_encoded, X_test_encoded = encode_categorical(X_train, X_test)

    # Convert to NumPy
    X_train_num = X_train_encoded.to_numpy(dtype=float)
    X_test_num = X_test_encoded.to_numpy(dtype=float)

    y_train_num = y_train.to_numpy(dtype=int)
    y_test_num = y_test.to_numpy(dtype=int)

    # Calculate training statistics
    train_mean = calculate_mean(X_train_num)
    train_std = calculate_std(X_train_num, train_mean)

    # Standardize
    X_train_scaled = standardize(X_train_num, train_mean, train_std)

    X_test_scaled = standardize(X_test_num, train_mean, train_std)

    # Create model
    model = LogisticRegression(learning_rate=0.1, epochs=1000)

    # Train
    model.fit(X_train_scaled, y_train_num)

    # Predict probabilities
    probabilities = predict_probability(X_test_scaled, model.weights, model.bias)

    # Convert probabilities to classes
    predictions = probability_to_class(probabilities)

    # Calculate metrics
    results = {
        "accuracy": accuracy(y_test_num, predictions),
        "precision": precision(y_test_num, predictions),
        "recall": recall(y_test_num, predictions),
        "f1_score": f1_score(y_test_num, predictions),
    }

    return model, results
