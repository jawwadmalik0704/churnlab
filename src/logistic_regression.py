import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def predict_probability(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)

def probability_to_class(probabilities, threshold=0.5):
    # If probability is greater than or equal to the threshold, predict class 1, Otherwise, predict class 0.
    return (probabilities >= threshold).astype(int)