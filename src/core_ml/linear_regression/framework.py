from sklearn.linear_model import LinearRegression


class SklearnLinearRegression:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        """
        Fit the linear regression model using scikit-learn's implementation.
        X.shape: (m,n) -> m : number of samples, n : number of features
        y.shape: (m,) -> m : number of results
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Return the predicted value with trained parameters for the given input X.
        """
        return self.model.predict(X)
