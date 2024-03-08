import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.label_map = None
        self.loss_history = []  # To store the loss at each iteration


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def log_loss(self, y_true, y_pred):
        """
        Computes the binary cross-entropy loss.
        """
        epsilon = 1e-15  # To prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def gradient_descent_step(self, X, y):
        """
        Performs one step of gradient descent updating the weights and bias.
        """
        n_samples, n_features = X.shape
        model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(model)

        dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
        db = (1 / n_samples) * np.sum(predictions - y)

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        return predictions  # Return predictions for loss calculation

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

        for _ in range(self.n_iterations):
            predictions = self.gradient_descent_step(X, y)
            loss = self.log_loss(y, predictions)
            self.loss_history.append(loss)

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(model)
        predicted_labels = [1 if i > 0.5 else 0 for i in predictions]
        
        # Map predicted labels back to original class labels
        inverse_label_map = {v: k for k, v in self.label_map.items()}
        original_labels = np.vectorize(inverse_label_map.get)(predicted_labels)
        
        return original_labels
    
class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.label_map = None
        self.loss_history = []  # To store the loss at each iteration
        self.name = "Softmax Regression"
    
    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)
    
    def compute_loss(self, y_true, y_pred):
        """
        Computes the categorical cross-entropy loss.
        """
        epsilon = 1e-15  # To prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred))
    
    def gradient_descent_step(self, X, y_one_hot, probabilities):
        """
        Performs one step of gradient descent updating the weights and bias.
        """
        n_samples = X.shape[0]
        # Compute the gradient on scores
        dw = (1 / n_samples) * np.dot(X.T, (probabilities - y_one_hot))
        db = (1 / n_samples) * np.sum(probabilities - y_one_hot, axis=0, keepdims=True)
        
        # Update parameters
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    
    def fit(self, X, y, print_loss = False):

        # Extract unique classes and sort them to ensure consistency
        unique_labels = np.unique(y)
        self.label_map = {original_label: i for i, original_label in enumerate(unique_labels)}

        # Map the labels to [0, n_classes-1]
        mapped_y = np.vectorize(self.label_map.get)(y)
        
        n_samples, n_features = X.shape
        n_classes = len(unique_labels)

        
        # Initialize weights and bias
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))
        self.loss_history = []  # Reset loss history for each fit call
        
        # One-hot encode y
        y_one_hot = np.eye(n_classes)[mapped_y]

        for i in range(self.n_iterations):
            # Compute the linear model predictions
            scores = np.dot(X, self.weights) + self.bias
            
            # Apply softmax to obtain class probabilities
            probabilities = self.softmax(scores)
            
            # Compute the loss and store it
            loss = self.compute_loss(y_one_hot, probabilities)
            self.loss_history.append(loss)
            
            # Perform a gradient descent step
            self.gradient_descent_step(X, y_one_hot, probabilities)
            
            # Print the loss every 100 iterations
            if i % 100 == 0 and print_loss:
                print(f"Iteration {i}: Loss {loss}")
    
    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        probabilities = self.softmax(scores)
        predicted_classes = np.argmax(probabilities, axis=1)
        
        # Map predicted labels back to original class labels
        inverse_label_map = {i: original_label for original_label, i in self.label_map.items()}
        original_labels = np.vectorize(inverse_label_map.get)(predicted_classes)
        
        return original_labels


