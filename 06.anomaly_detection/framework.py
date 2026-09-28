import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state

        self.model = IsolationForest(
            contaminationm=self.contamination, random_state=self.random_state
        )

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return 1 if the sample is an anomaly, 0 if it is normal.
        """
        predictions = self.model.predict(X)
        return (predictions == -1).astype(int)
