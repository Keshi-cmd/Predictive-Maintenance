import numpy as np
import pandas as pd

# This file contains all the models (use sklearn this file is only for learning purpose)

# # Model function
# def model(x_train, w = None, b = None):
#     m, n = x_train.shape
#     if (w == None) & (b == None):
#         w =  np.zeros((n,))
#         b = 0
#     f_wb = np.dot(x_train, w) + b
#     return f_wb

# # Sigmoid Function
# def sigmoid(z):
#     g = 1 / (np.exp(-z) + 1)
#     return g

# Logistic Regression
class logistic_regression:
    def __init__(self, learning_rate, iterations, w = None, b = None):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = w
        self.b = b
        self.cost_hist = None

    def fit(self, x_train, y_train):
        m, n = x_train.shape
        if (self.w is None) and (self.b is None):
            self.w = np.zeros((n,))
            self.b = 0

        self.cost_hist = []
        for epoch in range(self.iterations):
            z = np.dot(x_train, self.w) + self.b
            g = 1 / (np.exp(-z) + 1)
            loss = np.sum(y_train * np.log(g) + (1 - y_train) * np.log(1-g))
            J_wb = -1 / m * loss
            cost = J_wb

            J_dw = 1 / m * x_train.T @ (g - y_train)
            J_db = np.sum(1 / m * (g - y_train))
            self.w = self.w - self.learning_rate * J_dw
            self.b = self.b - self.learning_rate * J_db
            
            if epoch % 100 == 0 or epoch == self.iterations - 1:
                self.cost_hist.append(J_wb)
                print(f"Epoch {epoch} cost: {cost}")

    def predict_prob(self, X):
        z = np.dot(X, self.w) + self.b
        p = 1 / (np.exp(-z) + 1)
        return p

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        p = 1 / (np.exp(-z) + 1)
        y = (p > 0.5).astype(int)
        return y