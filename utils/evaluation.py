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

def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if (cm[1,1] + cm[0,1]) == 0:
        p = 0
    else:
        p = cm[1,1] / (cm[1,1] + cm[0,1])
    return p

def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    r = cm[1,1] / (cm[1,1] + cm[1,0])
    return r

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if (p + r) == 0:
        f1 = 0
    else:
        f1 = (2*p*r) / (p + r)
    return f1

def false_positive_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if (cm[0,1]+cm[0,0]) == 0:
        fpr = 0
    else:
        fpr = cm[0,1] / (cm[0,1] + cm[0,0])
    return fpr
