import numpy as np
import pandas as pd


def confusion_matrix(y_true, y_pred):
    TP = ((y_pred == 1) & (y_pred == y_true)).sum()
    TN = ((y_pred == 0) & (y_pred == y_true)).sum()
    FP = ((y_pred == 1) & (y_pred != y_true)).sum()
    FN = ((y_pred == 0) & (y_pred != y_true)).sum()

    cm = np.array([[TN, FP],
                  [FN, TP]])
    return cm

def _get_values(cm):

    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TP = cm[1,1]

    return TP, TN, FP, FN

def precision(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    TP, TN, FP, FN = _get_values(cm)

    if (TP + FP) == 0:
        return 0

    return TP / (TP + FP)

def recall(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    TP, TN, FP, FN = _get_values(cm)

    if (TP + FN) == 0:
        return 0

    return TP / (TP + FN)

def f1_score(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    TP, TN, FP, FN = _get_values(cm)

    if (2 * TP + FP + FN) == 0:
        return 0

    return (2 * TP) / (2 * TP + FP + FN)

def false_positive_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if (cm[0,1]+cm[0,0]) == 0:
        fpr = 0
    else:
        fpr = cm[0,1] / (cm[0,1] + cm[0,0])
    return fpr

def auc(fpr, tpr):
    auc_value = 0
    order = np.argsort(fpr)
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]

    for i in range(len(fpr_sorted) - 1):
        width = fpr_sorted[i+1] - fpr_sorted[i]
        height = (tpr_sorted[i] + tpr_sorted[i+1]) / 2
        auc_value += width * height

    return auc_value

