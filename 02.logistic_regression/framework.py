from sklearn.linear_model import LogisticRegression


class SklearnLogisticRegression:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        """
        Fit the logistic regression model using scikit-learn's implementation.
        X.shape: (m,n) -> m : number of samples, n : number of features
        y.shape: (m,) -> m : number of results
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Return the predicted class labels (0 or 1) for the given input X.
        """
        return self.model.predict(X)
