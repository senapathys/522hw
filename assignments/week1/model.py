import numpy as np


class LinearRegression:
    """
    Class for linear regression model
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = np.ndarray([])
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit linear regression to input data
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        weights = np.linalg.inv(X.T @ X) @ X.T @ y
        self.w = weights[1:]
        self.b = weights[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict output for input data
        """
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    Class for linear regression model using gradient descent
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fit linear regression to input data
        """
        num_samples, num_features = X.shape

        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.w = np.zeros(num_features + 1)
        self.b = 0

        for _ in range(epochs):
            y_pred = X @ self.w + self.b
            residuals = y - y_pred
            grad_w = -(2 / num_samples) * X.T @ residuals
            grad_b = -(2 / num_samples) * np.sum(residuals)
            self.b -= lr * grad_b
            self.w -= lr * grad_w

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict output for input data
        """
        if X.shape[1] < self.w.shape[0]:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.w + self.b
