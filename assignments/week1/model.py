import numpy as np


class LinearRegression:
    w: np.ndarray
    b: float

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # add bias column
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # add bias column
        return X @ np.append(self.intercept_, self.coef_)


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # add bias column
        n_samples, n_features = X.shape
        self.coef_ = np.random.randn(n_features)
        self.intercept_ = np.random.randn()

        for _ in range(epochs):
            y_pred = X @ self.coef_ + self.intercept_
            residuals = y - y_pred
            gradient_coef = -(2 / n_samples) * X.T @ residuals
            gradient_intercept = -(2 / n_samples) * np.sum(residuals)
            print(y_pred)
            self.coef_ -= lr * gradient_coef
            self.intercept_ -= lr * gradient_intercept

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # add bias column
        return X @ self.coef_ + self.intercept_
