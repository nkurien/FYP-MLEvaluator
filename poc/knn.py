import numpy as np
class KNearestNeighbours:
    """
    K-Nearest Neighbours algorithm for classification.

    Attributes:
        k (int): Number of nearest neighbours to consider for making predictions.
        X_training_data (array-like): Training data features.
        y_training_labels (array-like): Training data labels.
    """
    def __init__(self, k=1):
        """
        Constructor for KNearestNeighbours class.

        Args:
            k (int, optional): Number of neighbours to consider. Defaults to 1.
        """
        self.k = k
        self.X_training_data = None
        self.y_training_labels = None

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.

        Args:
            X (array-like): Training data, a 2D array of shape (n_samples, n_features).
            y (array-like): Target values, a 1D array of shape (n_samples,).

        Raises:
            ValueError: If k is greater than the number of training samples.
        """
        if self.k > len(X):
            raise ValueError("k must be less than or equal to the number of training data points.")
        self.X_training_data = X
        self.y_training_labels = y

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Args:
            X (array-like): Test data, a 2D array of shape (n_samples, n_features).

        Returns:
            list: Predicted class labels for each data sample.
        """
        predictions = [self._predict_point(point) for point in X]
        return predictions

    def _predict_point(self, point):
        """
        Predict the class label for a single data point.

        Args:
            point (array-like): A single data sample.

        Returns:
            The predicted class label.
        """
        k_nearest = [(-1, float('inf')) for _ in range(self.k)]  # [(index, distance), ...]

        for i, training_point in enumerate(self.X_training_data):
            if (point == training_point).all():  # Skip the identical point
                continue

            distance = self._euclidean_distance(point, training_point)
            
            # Check if the distance is smaller than the current k-nearest distances
            for j, (idx, dist) in enumerate(k_nearest):
                if distance < dist:
                    k_nearest.insert(j, (i, distance))
                    k_nearest = k_nearest[:self.k]  # Keep only the k-nearest distances
                    break

        # Get labels and distances of the k-nearest points
        k_labels_distances = [(self.y_training_labels[idx], dist) for idx, dist in k_nearest]

        # Count the occurrences of each label and store minimum distance for each label
        label_counts = {}
        for label, dist in k_labels_distances:
            if label not in label_counts:
                label_counts[label] = [0, float('inf')]
            label_counts[label][0] += 1  # Increase count
            label_counts[label][1] = min(label_counts[label][1], dist)  # Store the minimum distance

        # Find the label with the maximum count and if there's a tie, choose the one with the smallest distance
        majority_label = max(label_counts, key=lambda x: (label_counts[x][0], -label_counts[x][1]))

        return majority_label

    def _euclidean_distance(self, p1, p2):
         """
        Calculate the Euclidean distance between two points.

        Args:
            p1 (array-like): First point.
            p2 (array-like): Second point.

        Returns:
            float: Euclidean distance between p1 and p2."""
         
         return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))