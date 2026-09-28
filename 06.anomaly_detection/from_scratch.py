import numpy as np


class AnomalyDetector:
    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("The model has not been fitted yet.")

        z_scores = (X - self.mean) / (self.std + self.epsilon)
        probabilities = 1 - np.exp(-0.5 * z_scores**2)

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities < self.epsilon).astype(int)
