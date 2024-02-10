import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.label_map = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Check for more than two unique labels
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("Logistic Regression is a binary classifier and requires exactly 2 classes.")
        
        # Store mapping of unique labels to 0 and 1
        self.label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        mapped_y = np.vectorize(self.label_map.get)(y)
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - mapped_y))
            db = (1 / n_samples) * np.sum(predictions - mapped_y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(model)
        predicted_labels = [1 if i > 0.5 else 0 for i in predictions]
        
        # Map predicted labels back to original class labels
        inverse_label_map = {v: k for k, v in self.label_map.items()}
        original_labels = np.vectorize(inverse_label_map.get)(predicted_labels)
        
        return original_labels
