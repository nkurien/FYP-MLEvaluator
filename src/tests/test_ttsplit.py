import unittest
import numpy as np
import sys
sys.path.append("..")
from data_processing.train_test_split import train_test_split  # Adjust the import according to your project structure

class TestTrainTestSplit(unittest.TestCase):

    def setUp(self):
        # Sample dataset for testing
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([0, 1, 0, 1])

    def test_proportional_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25)
        self.assertEqual(len(X_train), 3)
        self.assertEqual(len(X_test), 1)
        self.assertEqual(len(y_train), 3)
        self.assertEqual(len(y_test), 1)

    def test_absolute_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=2)
        self.assertEqual(len(X_train), 2)
        self.assertEqual(len(X_test), 2)
        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(y_test), 2)

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            train_test_split([], [], test_size=0.25)

    def test_unequal_lengths(self):
        with self.assertRaises(ValueError):
            train_test_split(self.X, np.array([0, 1]), test_size=0.25)

    def test_invalid_test_size(self):
        with self.assertRaises(ValueError):
            train_test_split(self.X, self.y, test_size=1.5)

    def test_single_sample(self):
        X_single = np.array([[1, 2]])
        y_single = np.array([0])
        X_train, X_test, y_train, y_test = train_test_split(X_single, y_single, test_size=0.5)
        self.assertEqual(len(X_train), 1)
        self.assertEqual(len(X_test), 0)
        self.assertEqual(len(y_train), 1)
        self.assertEqual(len(y_test), 0)

    def test_shuffle_consistency_with_seed(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, seed=42)
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(self.X, self.y, test_size=0.25, seed=42)
        np.testing.assert_array_equal(X_train, X_train_2)
        np.testing.assert_array_equal(X_test, X_test_2)
    
    def test_large_test_size_integer(self):
        with self.assertRaises(ValueError):
            train_test_split(self.X, self.y, test_size=100)

    def test_invalid_test_size_type(self):
        with self.assertRaises(ValueError):
            train_test_split(self.X, self.y, test_size="invalid_type")
    
    def test_single_feature_data(self):
        X_single_feature = np.array([[1], [3], [5], [7]])
        X_train, X_test, y_train, y_test = train_test_split(X_single_feature, self.y, test_size=0.25)
        self.assertEqual(X_train.shape[1], 1)
        self.assertEqual(X_test.shape[1], 1)
    
    def test_list_input_data(self):
        X_list = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y_list = [0, 1, 0, 1]
        X_train, X_test, y_train, y_test = train_test_split(X_list, y_list, test_size=0.25)
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)
    
    def test_seed_with_large_dataset(self):
        X_large = np.random.rand(100, 10)
        y_large = np.random.randint(2, size=100)
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_large, y_large, test_size=0.25, seed=42)
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_large, y_large, test_size=0.25, seed=42)
        np.testing.assert_array_equal(X_train_1, X_train_2)
    
    def test_shuffle_different_seeds(self):
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(self.X, self.y, test_size=0.25, seed=42)
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(self.X, self.y, test_size=0.25, seed=24)
        self.assertFalse(np.array_equal(X_train_1, X_train_2))

    def test_same_seed_different_data(self):
        X_new = np.array([[9, 10], [11, 12], [13, 14], [15, 16]])
        y_new = np.array([1, 0, 1, 0])
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(self.X, self.y, test_size=0.25, seed=42)
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_new, y_new, test_size=0.25, seed=42)
        self.assertFalse(np.array_equal(X_train_1, X_train_2))

    def test_seed_with_different_test_sizes(self):
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(self.X, self.y, test_size=0.25, seed=42)
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(self.X, self.y, test_size=0.5, seed=42)
        self.assertNotEqual(len(X_train_1), len(X_train_2))







if __name__ == '__main__':
    unittest.main()
