import numpy as np


class LinearRegression:
    w: np.ndarray
    b: float

    def __init__(self):
        self.weights = None
        self.bias = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        weights = np.linalg.inv(X.T @ X) @ X.T @ y
        self.weights = weights[1:]
        self.bias = weights[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000):
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        self.weights = np.zeros(X.shape[1]-1)
        self.bias = 0

        for _ in range(epochs):
            print(self.weights)
            print(self.bias)
            y_pred = X @ np.hstack((self.bias, self.weights))
            residuals = y_pred - y
            gradient = X.T @ residuals
            self.bias -= lr * gradient[0]
            self.weights -= lr * gradient[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias
