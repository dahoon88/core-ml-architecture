import numpy as np


class NeuralNetwork:
    """
    Simple 2 - layer Neutral Network (MLP) built form scratch
    Input Layer -> Hidden Layer (ReLU) -> Output Layer (Sigmoid)
    """

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

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

        self.loss_history = []

    def relu(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)

    def relu_derivative(self, Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(float)

    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-Z))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the neural network using forward propagation and backpropagation.
        X.shape: (n_samples, input_size)
        y.shape: (n_samples, output_size = 1)
        """
        m = X.shape[0]
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        epsilon = 1e-15

        for _ in range(self.n_iterations):
            # Forward propagation
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.relu(Z1)

            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.sigmoid(Z2)

            # Compute Loss (Binary Cross-Entropy Loss)
            loss = (-1 / m) * np.sum(
                y * np.log(A2 + epsilon) + (1 - y) * np.log(1 - A2 + epsilon)
            )
            self.loss_history.append(loss)

            # Backpropagation
            dZ2 = A2 - y
            dW2 = (1 / m) * np.dot(A1.T, dZ2)
            db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.relu_derivative(Z1)
            dW1 = (1 / m) * np.dot(X.T, dZ1)
            db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

            # Update weights and biases
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return the predicted probabilities for the given input X.
        """
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        return A2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the predicted class labels for the given input X.
        """
        return (self.predict_proba(X) > 0.5).astype(int)
