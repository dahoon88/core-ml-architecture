import numpy as np


class AnomalyDetector:
    """
    Gaussian Distribution-based Anomaly Detection built from scratch using vectorized NumPy.
    Computes multivariate Gaussian probability density (assuming feature independence)
    and flags samples with density below threshold epsilon as anomalies.
    """

    def __init__(self, epsilon: float = 1e-4):
        self.epsilon = epsilon
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        """
        Fit the Gaussian distribution parameters (mean and standard deviation) on training data.
        X.shape: (m, n) -> m: samples, n: features
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute joint probability density p(x) for each sample assuming feature independence.
        Returns array of shape (m,).
        """
        if self.mean is None or self.std is None:
            raise ValueError("The model has not been fitted yet.")

        # Prevent division by zero with variance floor
        var = (self.std**2) + 1e-9
        exponent = -0.5 * ((X - self.mean) ** 2) / var

        # Log density for numerical stability across multiple features
        log_density = -0.5 * np.log(2 * np.pi * var) + exponent
        total_log_density = np.sum(log_density, axis=1)

        return np.exp(total_log_density)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Flag samples with joint probability density below epsilon as anomalies (1 = anomaly, 0 = normal).
        Returns binary array of shape (m,).
        """
        probabilities = self.predict_proba(X)
        return (probabilities < self.epsilon).astype(int)
