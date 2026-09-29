import numpy as np


class MatrixFactorization:
    """
    Collaborative Filtering via Matrix Factorization using Gradient Descent.
    Learns user and item latent feature matrices from a sparse rating matrix.
    """

    def __init__(
        self,
        n_factors: int = 5,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: float = 0.02,
    ):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization

        self.user_matrix = None
        self.item_matrix = None

    def fit(self, R: np.ndarray):
        n_users, n_items = R.shape
        self.user_matrix = np.random.normal(
            scale=1.0 / self.n_factors, size=(n_users, self.n_factors)
        )
        self.item_matrix = np.random.normal(
            scale=1.0 / self.n_factors, size=(n_items, self.n_factors)
        )
        mask = R > 0

        for _ in range(self.n_iterations):
            R_pred = np.dot(self.user_matrix, self.item_matrix.T)

            error = mask * (R - R_pred)

            user_grad = (
                -2 * np.dot(error, self.item_matrix)
                + 2 * self.regularization * self.user_matrix
            )
            item_grad = (
                -2 * np.dot(error.T, self.user_matrix)
                + 2 * self.regularization * self.item_matrix
            )

            self.user_matrix -= self.learning_rate * user_grad
            self.item_matrix -= self.learning_rate * item_grad

        return self

    def predict(self) -> np.ndarray:
        return np.dot(self.user_matrix, self.item_matrix.T)
