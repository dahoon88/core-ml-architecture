import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class SklearnDecisionTree:
    def __init__(self, max_depth=None, random_state=42):
        """
        Initialize the Scikit-Learn Decision Tree model.
        """
        self.model = DecisionTreeClassifier(
            max_depth=max_depth, criterion="entropy", random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the decision tree model using scikit-learn's implementation.
        X.shape: (m,n) -> m : number of samples, n : number of features
        y.shape: (m,) -> m : number of results
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the predicted class labels for the given input X.
        """
        return self.model.predict(X)


class SklearnRandomForest:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize the Scikit-Learn Random Forest ensemble model.
        """
        # n_estimators: The number of trees in the forest
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion="entropy",
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the random forest model using scikit-learn's implementation.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the predicted class labels for the given input X.
        """
        return self.model.predict(X)
