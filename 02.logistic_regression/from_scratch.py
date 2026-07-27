import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        # Learning rate and iterations for gradient descent
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # Parameters
        self.weights = None
        self.bias = None

        # Cost history for each iteration
        self.cost_history = []

    def sigmoid(self, z):
        """
        Compute the sigmoid function for the given input z.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Get the data X and result y and train model with gradient descent.
        X.shape: (m,n) -> m : number of samples, n : number of features
        y.shape: (m,) -> m : number of results
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        epsilon = 1e-15  # To avoid log(0)
        for i in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(z)

            # Error
            error = y_predicted - y

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute cost
            cost = (-1 / n_samples) * np.sum(
                y * np.log(y_predicted + epsilon)
                + (1 - y) * np.log(1 - y_predicted + epsilon)
            )
            self.cost_history.append(cost)

    def predict_proba(self, X):
        """
        Return the predicted probabilities with trained parameters for the given input X.
        """
        return self.sigmoid(X.dot(self.weights) + self.bias)

    def predict(self, X, threshold=0.5):
        """
        Return the predicted class labels (0 or 1) based on the predicted probabilities and a given threshold.
        """
        return (self.predict_proba(X) >= threshold).astype(int)
