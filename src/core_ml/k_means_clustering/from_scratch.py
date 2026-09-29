import numpy as np


class KMeans:
    """
    K-Means Clustering algorithm built from scratch using vectorized NumPy operations.
    """

    def __init__(self, n_clusters: int = 3, max_iter: int = 100, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X: np.ndarray):
        m, n = X.shape

        # Initialize centroids ramdonly
        random_idxs = np.random.choice(m, self.n_clusters, replace=False)
        self.centroids = X[random_idxs]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.zeros((self.n_clusters, n))
            for i in range(self.n_clusters):
                if np.any(labels == i):
                    new_centroids[i] = X[labels == i].mean(axis=0)
                else:
                    new_centroids[i] = self.centroids[i]
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

        return self

    def predict(self, X: np.ndarray):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
