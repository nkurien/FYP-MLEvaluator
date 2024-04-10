import numpy as np

class LogisticRegression:
    """
    Logistic Regression classifier.

    Parameters:
    -----------
    learning_rate : float, default=0.01
        The learning rate for gradient descent optimization.
    n_iterations : int, default=1000
        The number of iterations to run gradient descent.

    Attributes:
    -----------
    weights : numpy.ndarray
        The learned weights of the logistic regression model.
    bias : float
        The learned bias term of the logistic regression model.
    label_map : dict
        Mapping of original class labels to 0 and 1.
    loss_history : list
        History of the loss values during training.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.label_map = None
        self.loss_history = []  # To store the loss at each iteration

    def sigmoid(self, z):
        """
        Compute the sigmoid activation function.

        Parameters:
        -----------
        z : numpy.ndarray
            The input to the sigmoid function.

        Returns:
        --------
        numpy.ndarray
            The output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
    
    def log_loss(self, y_true, y_pred):
        """
        Compute the binary cross-entropy loss.

        Parameters:
        -----------
        y_true : numpy.ndarray
            The true labels.
        y_pred : numpy.ndarray
            The predicted probabilities.

        Returns:
        --------
        float
            The binary cross-entropy loss.
        """
        epsilon = 1e-15  # To prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def gradient_descent_step(self, X, y):
        """
        Perform one step of gradient descent to update the weights and bias.

        Parameters:
        -----------
        X : numpy.ndarray
            The input features.
        y : numpy.ndarray
            The true labels.

        Returns:
        --------
        numpy.ndarray
            The predicted probabilities.
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
        """
        Fit the logistic regression model to the training data.

        Parameters:
        -----------
        X : numpy.ndarray
            The input features.
        y : numpy.ndarray
            The target labels.

        Raises:
        -------
        ValueError
            If the number of unique labels is not equal to 2.
        """
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
            predictions = self.gradient_descent_step(X, mapped_y)
            loss = self.log_loss(mapped_y, predictions)
            self.loss_history.append(loss)

    def predict(self, X):
        """
        Make predictions using the trained logistic regression model.

        Parameters:
        -----------
        X : numpy.ndarray
            The input features.

        Returns:
        --------
        numpy.ndarray
            The predicted class labels.
        """
        model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(model)
        predicted_labels = [1 if i > 0.5 else 0 for i in predictions]
        
        # Map predicted labels back to original class labels
        inverse_label_map = {v: k for k, v in self.label_map.items()}
        original_labels = np.vectorize(inverse_label_map.get)(predicted_labels)
        
        return original_labels
    
class SoftmaxRegression:
    """
    Softmax Regression classifier.

    Parameters:
    -----------
    learning_rate : float, default=0.01
        The learning rate for gradient descent optimization.
    n_iterations : int, default=1000
        The number of iterations to run gradient descent.
    lambda_ : float, default=0.01
        The regularization strength.

    Attributes:
    -----------
    weights : numpy.ndarray
        The learned weights of the softmax regression model.
    bias : numpy.ndarray
        The learned bias terms of the softmax regression model.
    label_map : dict
        Mapping of original class labels to integers.
    loss_history : list
        History of the loss values during training.
    name : str
        The name of the classifier.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.label_map = None
        self.loss_history = []  # To store the loss at each iteration
        self.name = "Softmax Regression"
        self.lambda_ = lambda_  # Regularization strength

    def softmax(self, z):
        """
        Compute the softmax activation function.

        Parameters:
        -----------
        z : numpy.ndarray
            The input to the softmax function.

        Returns:
        --------
        numpy.ndarray
            The output of the softmax function.
        """
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute the categorical cross-entropy loss with L2 regularization.

        Parameters:
        -----------
        y_true : numpy.ndarray
            The true one-hot encoded labels.
        y_pred : numpy.ndarray
            The predicted probabilities.

        Returns:
        --------
        float
            The categorical cross-entropy loss with L2 regularization.
        """
        epsilon = 1e-15  # To prevent log(0)
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        cross_entropy_loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        l2_penalty = (self.lambda_ / 2) * np.sum(np.square(self.weights))
        total_loss = cross_entropy_loss + l2_penalty
        return total_loss

    def gradient_descent_step(self, X, y_one_hot, probabilities):
        """
        Perform one step of gradient descent to update the weights and bias.

        Parameters:
        -----------
        X : numpy.ndarray
            The input features.
        y_one_hot : numpy.ndarray
            The true one-hot encoded labels.
        probabilities : numpy.ndarray
            The predicted probabilities.
        """
        n_samples = X.shape[0]
        dw = (1 / n_samples) * np.dot(X.T, (probabilities - y_one_hot))
        db = (1 / n_samples) * np.sum(probabilities - y_one_hot, axis=0, keepdims=True)
        
        dw += self.lambda_ * self.weights
        
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(self, X, y, print_loss=False):
        """
        Fit the softmax regression model to the training data.

        Parameters:
        -----------
        X : numpy.ndarray
            The input features.
        y : numpy.ndarray
            The target labels.
        print_loss : bool, default=False
            Whether to print the loss every 100 iterations.
        """
        unique_labels = np.unique(y)
        self.label_map = {original_label: i for i, original_label in enumerate(unique_labels)}

        mapped_y = np.vectorize(self.label_map.get)(y)
        
        n_samples, n_features = X.shape
        n_classes = len(unique_labels)

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))
        self.loss_history = []  # Reset loss history for each fit call
        
        y_one_hot = np.eye(n_classes)[mapped_y]

        for i in range(self.n_iterations):
            scores = np.dot(X, self.weights) + self.bias
            probabilities = self.softmax(scores)
            
            loss = self.compute_loss(y_one_hot, probabilities)
            self.loss_history.append(loss)
            
            self.gradient_descent_step(X, y_one_hot, probabilities)
            
            if i % 100 == 0 and print_loss:
                print(f"Iteration {i}: Loss {loss}")
    
    def predict(self, X):
        """
        Make predictions using the trained softmax regression model.

        Parameters:
        -----------
        X : numpy.ndarray
            The input features.

        Returns:
        --------
        numpy.ndarray
            The predicted class labels.
        """
        scores = np.dot(X, self.weights) + self.bias
        probabilities = self.softmax(scores)
        predicted_classes = np.argmax(probabilities, axis=1)
        
        inverse_label_map = {i: original_label for original_label, i in self.label_map.items()}
        original_labels = np.vectorize(inverse_label_map.get)(predicted_classes)
        
        return original_labels