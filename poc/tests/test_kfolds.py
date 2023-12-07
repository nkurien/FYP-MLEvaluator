import unittest
import numpy as np
import sys
sys.path.append("..")
from cross_validation import _k_folds, k_folds_accuracy_scores, k_folds_accuracy_score
from classification_tree import ClassificationTree
from knn import KNearestNeighbours

class MockModel:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros(len(X))

class MockModelWithoutFit:
    def predict(self, X):
        return np.zeros(len(X))
    
class MockModelWithoutPredict:
    def fit(self, X, y):
        pass


class TestKFolds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate a mock dataset
        cls.X, cls.y = np.random.rand(100, 10), np.random.randint(0, 2, 100)
        cls.knn = KNearestNeighbours(k=3)
        cls.tree = ClassificationTree(max_depth=5, min_size=2)
    
    def test_empty_dataset(self):
        empty_X, empty_y = np.array([]), np.array([])
        with self.assertRaises(ValueError):
            _k_folds(empty_X, empty_y, 5)

    def test_single_fold(self):
        with self.assertRaises(ValueError):
            scores = k_folds_accuracy_scores(self.knn, self.X, self.y, k=1)
        
    

    def test_seed_reproducibility(self):
        seed = 42
        folds1 = _k_folds(self.X, self.y, 5, seed)
        folds2 = _k_folds(self.X, self.y, 5, seed)
        for fold1, fold2 in zip(folds1, folds2):
            np.testing.assert_array_equal(fold1[0][0], fold2[0][0])  # X_train
            np.testing.assert_array_equal(fold1[0][1], fold2[0][1])  # y_train
            np.testing.assert_array_equal(fold1[1][0], fold2[1][0])  # X_test
            np.testing.assert_array_equal(fold1[1][1], fold2[1][1])  # y_test

    def test_model_fitting(self):
        # Test with KNearestNeighbours
        scores_knn = k_folds_accuracy_scores(self.knn, self.X, self.y, 5)
        self.assertTrue(len(scores_knn) == 5)

        # Test with ClassificationTree
        scores_tree = k_folds_accuracy_scores(self.tree, self.X, self.y, 5)
        self.assertTrue(len(scores_tree) == 5)

    def test_varying_k_for_knn(self):
        for k in [1, 5, 10]:
            knn_model = KNearestNeighbours(k=k)
            scores = k_folds_accuracy_scores(knn_model, self.X, self.y, 5)
            self.assertTrue(len(scores) == 5)  # Check if scores are obtained for each fold
    
    def test_k_folds_float_input(self):
        with self.assertRaises(ValueError):
            scores = k_folds_accuracy_scores(self.knn, self.X, self.y, k=2.5)
    
    def test_model_without_fit(self):
        model_without_fit = MockModelWithoutFit()
        with self.assertRaises(AttributeError):
            k_folds_accuracy_scores(model_without_fit, self.X, self.y, 5)

    def test_model_without_predict(self):
        model_without_predict = MockModelWithoutPredict()
        with self.assertRaises(AttributeError):
            k_folds_accuracy_scores(model_without_predict, self.X, self.y, 5)


if __name__ == '__main__':
    unittest.main()

