import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        # Learning rate and iterations for gradient descent
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # Parameters
        self.weights = None
        self.bias = None

        # Cost history for each iteration
        self.cost_history = []

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
        for i in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Error
            error = y_predicted - y

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute cost
            cost = (1 / (2 * n_samples)) * np.sum(error**2)
            self.cost_history.append(cost)

    def predict(self, X):
        """
        Return the predicted value with trained parameters for the given input X.
        """
        return X.dot(self.weights) + self.bias
