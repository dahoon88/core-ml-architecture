import numpy as np
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.optim as optim


class SklearnNeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # Initialize the MLPClassifier from sklearn
        self.model = MLPClassifier(
            hidden_layer_sizes=(self.hidden_size,),
            activation="relu",
            solver="adam",
            learning_rate_init=self.learning_rate,
            max_iter=self.n_iterations,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class PyTorchNeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # Define the PyTorch model
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid(),
        )

        # Define the loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the PyTorch neural network using forward propagation and backpropagation.
        X.shape: (n_samples, input_size)
        y.shape: (n_samples, output_size = 1)
        """
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)

        for _ in range(self.n_iterations):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input X.
        X.shape: (n_samples, input_size)
        Returns: Predicted output (n_samples, output_size = 1)
        """
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input X.
        X.shape: (n_samples, input_size)
        Returns: Predicted output (n_samples, output_size = 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int).flatten()
