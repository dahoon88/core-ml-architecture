import numpy as np
from sklearn.cluster import KMeans as _SklearnKMeans


class SklearnKMeans:
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.model = _SklearnKMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_init="auto",
        )

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        self.centroids = self.model.cluster_centers_
        return self

    def predict(self, X: np.ndarray):
        return self.model.predict(X)
