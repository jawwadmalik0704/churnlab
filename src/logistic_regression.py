import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_probability(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)


def probability_to_class(probabilities, threshold=0.5):
    # If probability is greater than or equal to the threshold,
    # predict class 1. Otherwise, predict class 0.
    return (probabilities >= threshold).astype(int)


class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.loss_history = []

    def binary_cross_entropy(self, y_true, probabilities):
        epsilon = 1e-15

        probabilities = np.clip(probabilities, epsilon, 1 - epsilon) ##Keep every probability between epsilon and 1-epsilon

        loss = -np.mean(
            y_true * np.log(probabilities) + (1 - y_true) * np.log(1 - probabilities)
        )

        return loss

    def calculate_gradients(self, X, y, probabilities):
        errors = probabilities - y

        weight_gradient = np.dot(X.T, errors) / len(y)
        bias_gradient = np.mean(errors)

        return weight_gradient, bias_gradient

    def update_parameters(self, weight_gradient, bias_gradient):
        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []

        for epoch in range(self.epochs):

            # 1. Make predictions
            probabilities = predict_probability(X, self.weights, self.bias)

            # 2. Calculate loss
            loss = self.binary_cross_entropy(y, probabilities)
            self.loss_history.append(loss)
            # 3. Calculate gradients
            weight_gradient, bias_gradient = self.calculate_gradients(X, y, probabilities)

            # 4. Update parameters
            self.update_parameters(weight_gradient, bias_gradient)

            print(f"Epoch {epoch + 1}: Loss = {loss}")
