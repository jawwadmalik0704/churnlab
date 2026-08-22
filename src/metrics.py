import numpy as np


def confusion_matrix(y_true, y_pred):
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    true_positive = np.sum((y_true == 1) & (y_pred == 1))

    return true_negative, false_positive, false_negative, true_positive


def accuracy(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

    total = tn + fp + fn + tp

    if total == 0:
        return 0.0

    return (tp + tn) / total


def precision(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        return 0.0

    return 2 * (p * r) / (p + r)
