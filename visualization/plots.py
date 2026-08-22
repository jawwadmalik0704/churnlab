import matplotlib.pyplot as plt
from src.metrics import confusion_matrix


def plot_loss_curve(loss_history):
    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

    matrix = [[tn, fp], [fn, tp]]

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)

    plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])

    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, matrix[i][j], ha="center", va="center")

    plt.tight_layout()
    plt.show()


def plot_feature_weights(feature_names, weights):
    plt.figure(figsize=(9, 5))
    plt.bar(feature_names, weights)
    plt.xlabel("Features")
    plt.ylabel("Weight")
    plt.title("Learned Logistic Regression Feature Weights")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
