import numpy as np
from sklearn.tree import DecisionTreeRegressor

class BoostingTreeRegressor:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []
        self.weights = []

    def fit(self, X, y):
        n_samples = len(y)
        residuals = y.copy()
        sample_weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals, sample_weight=sample_weights)
            self.trees.append(tree)

            y_pred = tree.predict(X)
            residuals -= self.learning_rate * y_pred

            # Calculate loss and update sample weights
            loss = self.custom_loss(y, residuals)
            sample_weights = loss / np.sum(loss)

            self.weights.append(np.log(np.mean(loss) / np.mean(1 - loss)))

    def predict(self, X):
        predictions = np.zeros(len(X))
        for tree, weight in zip(self.trees, self.weights):
            predictions += weight * tree.predict(X)
        return predictions

    def custom_loss(self, y_true, y_pred):
        # Define your custom loss function here
        # For example, mean squared error:
        return np.square(y_true - y_pred)

# Example usage
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([1, 2, 3, 4, 5])

boosting_regressor = BoostingTreeRegressor(n_estimators=3, max_depth=1)
boosting_regressor.fit(X, y)

y_pred = boosting_regressor.predict(X)
print("Predictions:", y_pred)
