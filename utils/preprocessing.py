import numpy as np
import pandas as pd

class z_score:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x_train):
        m, n = x_train.shape
        x_new = np.zeros((m, n))
        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0)
        X_new = (x_train - self.mean) / self.std
        return X_new

    def transform(self, X_new):
        X_new = (X_new - self.mean) / self.std
        return X_new