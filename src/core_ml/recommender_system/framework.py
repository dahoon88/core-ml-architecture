import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PytorchRecommender(nn.Module):
    """
    Matrix Factorization Collaborative Filtering using PyTorch Embeddings.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 5,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
    ):
        super(PytorchRecommender, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # User and item embeddings
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def fit(self, R: np.ndarray):
        user_indices, item_indices = np.where(R > 0)
        ratings = R[user_indices, item_indices]

        user_tensor = torch.LongTensor(user_indices)
        item_tensor = torch.LongTensor(item_indices)
        ratings_tensor = torch.FloatTensor(ratings)

        for _ in range(self.n_iterations):
            self.optimizer.zero_grad()

            user_vecs = self.user_embeddings(user_tensor)
            item_vecs = self.item_embeddings(item_tensor)

            predictions = (user_vecs * item_vecs).sum(dim=1)

            loss = self.criterion(predictions, ratings_tensor)
            loss.backward()
            self.optimizer.step()

        return self

    def predict(self):
        with torch.no_grad():
            user_vecs = self.user_embeddings.weight
            item_vecs = self.item_embeddings.weight
            predictions = torch.matmul(user_vecs, item_vecs.t())
        return predictions.numpy()
