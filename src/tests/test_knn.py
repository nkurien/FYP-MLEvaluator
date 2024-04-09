import unittest
import sys
sys.path.append("..")
from models.knn import KNearestNeighbours
import numpy as np

class TestKNearestNeighbours(unittest.TestCase):

    def setUp(self):
        # Sample datasets for testing
        self.X_train = np.array([[1, 2], [3, 4], [5, 6]])
        self.y_train = np.array([0, 1, 1])
        self.X_test = np.array([[2, 3], [4, 5]])

    def test_initialization(self):
        knn = KNearestNeighbours(k=3)
        self.assertEqual(knn.k, 3)
        self.assertIsNone(knn.X_training_data)
        self.assertIsNone(knn.y_training_labels)

    def test_fit(self):
        knn = KNearestNeighbours(k=2)
        knn.fit(self.X_train, self.y_train)
        np.testing.assert_array_equal(knn.X_training_data, self.X_train)
        np.testing.assert_array_equal(knn.y_training_labels, self.y_train)

    def test_fit_invalid_k(self):
        knn = KNearestNeighbours(k=4)
        with self.assertRaises(ValueError):
            knn.fit(self.X_train, self.y_train)

    def test_predict(self):
        knn = KNearestNeighbours(k=1)
        knn.fit(self.X_train, self.y_train)
        predictions = knn.predict(self.X_test)
        expected = [0, 1]  # Based on the nearest neighbour
        self.assertEqual(predictions, expected)

    def test_predict_with_clear_tie(self):
        knn = KNearestNeighbours(k=2)
        
        # Revised training data 
        X_train = np.array([[1, 1], [7, 7], [8, 8]])
        y_train = np.array([0, 1, 1])
        knn.fit(X_train, y_train)

        # Test data point chosen such that it's closer to [1, 1] than to [7, 7]
        X_test = np.array([[2, 2]])

        expected = [0]

        # Make prediction and assert
        predictions = knn.predict(X_test)
        self.assertEqual(predictions, expected)


    def test_euclidean_distance(self):
        knn = KNearestNeighbours()
        distance = knn._euclidean_distance([1, 2], [4, 6])
        expected = np.sqrt((3**2) + (4**2))
        self.assertAlmostEqual(distance, expected)

    def test_predict_point(self):
        knn = KNearestNeighbours(k=2)
        knn.fit(self.X_train, self.y_train)
        prediction = knn._predict_point([2, 3])
        expected = 0  # Based on the nearest neighbours
        self.assertEqual(prediction, expected)
    
    def test_fit_with_different_dimensions(self):
        knn = KNearestNeighbours(k=1)
        X_train = np.array([[1, 2, 3], [4, 5, 6]])
        y_train = np.array([0, 1])
        knn.fit(X_train, y_train)
        np.testing.assert_array_equal(knn.X_training_data, X_train)
        np.testing.assert_array_equal(knn.y_training_labels, y_train)

    def test_predict_with_larger_dataset(self):
        knn = KNearestNeighbours(k=3)
        X_train_large = np.random.rand(100, 2)  # 100 points with 2 features each
        y_train_large = np.random.randint(0, 2, 100)  # Binary labels
        knn.fit(X_train_large, y_train_large)
        X_test = np.random.rand(10, 2)  # 10 test points
        predictions = knn.predict(X_test)
        self.assertIsInstance(predictions, list)  # Just check if predictions are returned as a list

    def test_predict_with_single_feature(self):
        knn = KNearestNeighbours(k=2)
        X_train = np.array([[1], [3], [5]])
        y_train = np.array([0, 0, 1])
        knn.fit(X_train, y_train)
        X_test = np.array([[2]])
        predictions = knn.predict(X_test)
        expected = [0]
        self.assertEqual(predictions, expected)
    
    def test_fit_with_zero_neighbours(self):
        with self.assertRaises(ValueError):
            KNearestNeighbours(k=0)
    
    def test_negative_k_value(self):
        with self.assertRaises(ValueError):
            KNearestNeighbours(k=-1)
    
    def test_non_integer_k_value(self):
        with self.assertRaises(ValueError):
            KNearestNeighbours(k=2.5)

    def test_identical_points_in_training_data(self):
        knn = KNearestNeighbours(k=2)
        X_train = np.array([[1, 1], [1, 1], [2, 2]])
        y_train = np.array([0, 0, 1])
        knn.fit(X_train, y_train)
        X_test = np.array([[1, 1]])
        predictions = knn.predict(X_test)
        expected = [0]  # Most common label among identical points
        self.assertEqual(predictions, expected)
    
    def test_empty_training_data(self):
        knn = KNearestNeighbours(k=1)
        with self.assertRaises(ValueError):
            knn.fit([], [])
    
    def test_mismatched_feature_dimensions(self):
        knn = KNearestNeighbours(k=1)
        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0, 1])
        knn.fit(X_train, y_train)
        X_test = np.array([[1, 2, 3]])
        with self.assertRaises(ValueError):
            knn.predict(X_test)
    
    def test_predict_without_fit(self):
        knn = KNearestNeighbours(k=1)
        X_test = np.array([[1, 2]])
        with self.assertRaises(ValueError):
            knn.predict(X_test)
    
    def test_large_k_value(self):
        knn = KNearestNeighbours(k=100)
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 1])
        with self.assertRaises(ValueError):
            knn.fit(X_train, y_train)
    
    def test_fit_with_unequal_lengths(self):
        knn = KNearestNeighbours(k=1)
        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0])
        with self.assertRaises(ValueError):
            knn.fit(X_train, y_train)













if __name__ == '__main__':
    unittest.main()
