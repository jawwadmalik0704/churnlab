import matplotlib.pyplot as plt
from src.metrics import confusion_matrix


def plot_loss_curve(loss_history):

    epochs = range(1, len(loss_history) + 1)

    fig, ax = plt.subplots(
        figsize=(10, 5),
        facecolor="white",
    )

    ax.set_facecolor("white")

    ax.plot(
        epochs,
        loss_history,
        linewidth=2.2,
        color="#2563EB",
    )

    ax.set_xlabel(
        "Epoch",
        color="#111827",
    )

    ax.set_ylabel(
        "Loss",
        color="#111827",
    )

    ax.set_title(
        "Training Loss vs Epoch",
        color="#111827",
        fontsize=14,
        fontweight="bold",
    )

    ax.tick_params(
        colors="#374151",
    )

    ax.grid(
        True,
        alpha=0.18,
        color="#64748B",
    )

    for spine in ax.spines.values():
        spine.set_color("#E5E7EB")

    fig.tight_layout()

    return fig


def plot_confusion_matrix(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    )

    matrix = [
        [tn, fp],
        [fn, tp],
    ]

    fig, ax = plt.subplots(
        figsize=(6, 5),
        facecolor="white",
    )

    ax.set_facecolor("white")

    image = ax.imshow(
        matrix,
        cmap="Blues",
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Predicted 0", "Predicted 1"],
        color="#111827",
    )

    ax.set_yticklabels(
        ["Actual 0", "Actual 1"],
        color="#111827",
    )

    ax.set_xlabel(
        "Prediction",
        color="#111827",
    )

    ax.set_ylabel(
        "Actual",
        color="#111827",
    )

    ax.set_title(
        "Confusion Matrix",
        color="#111827",
        fontsize=14,
        fontweight="bold",
    )

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                matrix[i][j],
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="#111827",
            )

    ax.tick_params(
        colors="#374151",
    )

    for spine in ax.spines.values():
        spine.set_color("#E5E7EB")

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    fig.tight_layout()

    return fig


def plot_feature_weights(feature_names, weights):

    fig, ax = plt.subplots(
        figsize=(10, 5),
        facecolor="white",
    )

    ax.set_facecolor("white")

    bars = ax.bar(
        feature_names,
        weights,
        color="#2563EB",
        alpha=0.9,
    )

    ax.axhline(
        0,
        color="#64748B",
        linewidth=1,
    )

    ax.set_xlabel(
        "Features",
        color="#111827",
    )

    ax.set_ylabel(
        "Weight",
        color="#111827",
    )

    ax.set_title(
        "Learned Logistic Regression Feature Weights",
        color="#111827",
        fontsize=14,
        fontweight="bold",
    )

    ax.tick_params(
        colors="#374151",
        axis="x",
        rotation=45,
    )

    ax.tick_params(
        colors="#374151",
        axis="y",
    )

    ax.grid(
        axis="y",
        alpha=0.15,
        color="#64748B",
    )

    for spine in ax.spines.values():
        spine.set_color("#E5E7EB")

    fig.tight_layout()

    return fig
