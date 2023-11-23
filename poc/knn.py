import numpy as np
class KNearestNeighbours:
    def __init__(self, k=1):
        self.k = k
        self.X_training_data = None
        self.y_training_labels = None

    def fit(self, X, y):
        if self.k > len(X):
            raise ValueError("k must be less than or equal to the number of training data points.")
        self.X_training_data = X
        self.y_training_labels = y

    def predict(self, X):
        predictions = [self._predict_point(point) for point in X]
        return predictions

    def _predict_point(self, point):
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

        # Get labels of the k-nearest points
        k_labels = [self.y_training_labels[idx] for idx, dist in k_nearest]
        
        # Majority vote
        majority_label = max(set(k_labels), key=k_labels.count)
        return majority_label

    def _euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))