import unittest
import numpy as np
import sys
sys.path.append('..')
from models.logistic_regression import LogisticRegression, SoftmaxRegression

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        # Set up any necessary data or objects for testing
        pass

    def test_initialization(self):
        lr = LogisticRegression(learning_rate=0.1, n_iterations=500)
        self.assertEqual(lr.learning_rate, 0.1)
        self.assertEqual(lr.n_iterations, 500)
        self.assertIsNone(lr.weights)
        self.assertIsNone(lr.bias)
        self.assertIsNone(lr.label_map)
        self.assertEqual(lr.loss_history, [])

    def test_sigmoid(self):
        lr = LogisticRegression()
        z = np.array([0, 1, -1, 10, -10])
        expected_output = np.array([0.5, 0.73105858, 0.26894142, 0.9999546, 4.5397868e-05])
        np.testing.assert_allclose(lr.sigmoid(z), expected_output, rtol=1e-6)

    def test_log_loss(self):
        lr = LogisticRegression()
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.8, 0.2, 0.7, 0.3])
        expected_loss = 0.2899092476264711
        self.assertAlmostEqual(lr.log_loss(y_true, y_pred), expected_loss)

    def test_gradient_descent_step(self):
        lr = LogisticRegression()
        X = np.array([[1, 2], [1, 3], [1, 5], [1, 7]])
        y = np.array([0, 1, 1, 0])
        lr.weights = np.zeros(X.shape[1])
        lr.bias = 0
        initial_predictions = lr.gradient_descent_step(X, y)
        self.assertIsNotNone(lr.weights)
        self.assertIsNotNone(lr.bias)
        # Ensure loss decreases after the gradient descent step
        initial_loss = lr.log_loss(y, initial_predictions)
        predictions_after_step = lr.gradient_descent_step(X, y)
        self.assertLess(lr.log_loss(y, predictions_after_step), initial_loss)


    def test_fit(self):
        lr = LogisticRegression()
        X = np.array([[1, 2], [1, 3], [1, 5], [1, 7]])
        y = np.array([0, 1, 1, 0])
        lr.fit(X, y)
        self.assertLess(lr.loss_history[-1], lr.loss_history[0])  # Check if loss decreased
        self.assertNotEqual(sum(lr.weights), 0)  # Ensure weights were updated
        self.assertNotEqual(lr.bias, 0)  # Ensure bias was updated

    def test_predict(self):
        lr = LogisticRegression()
        X = np.array([[1, 2], [1, 3], [1, 5], [1, 7]])
        y = np.array([0, 1, 1, 0])
        lr.fit(X, y)
        predictions = lr.predict(X)

        # Check if the predicted labels are within the set of unique labels in y
        unique_labels = np.unique(y)
        self.assertTrue(np.isin(predictions, unique_labels).all())

        # Check if the predicted labels have the same shape as y
        self.assertEqual(predictions.shape, y.shape)

class TestSoftmaxRegression(unittest.TestCase):
    def setUp(self):
        # Set up any necessary data or objects for testing
        pass

    def test_initialization(self):
        sr = SoftmaxRegression(learning_rate=0.1, n_iterations=500, lambda_=0.01)
        self.assertEqual(sr.learning_rate, 0.1)
        self.assertEqual(sr.n_iterations, 500)
        self.assertEqual(sr.lambda_, 0.01)
        self.assertIsNone(sr.weights)
        self.assertIsNone(sr.bias)
        self.assertIsNone(sr.label_map)
        self.assertEqual(sr.loss_history, [])

    def test_softmax(self):
        sr = SoftmaxRegression()
        z = np.array([[1, 2, 3], [4, 5, 6]])
        expected_output = np.array([[0.09003057, 0.24472847, 0.66524096],
                                    [0.09003057, 0.24472847, 0.66524096]])
        np.testing.assert_allclose(sr.softmax(z), expected_output, rtol=1e-6)

    def test_compute_loss(self):
        sr = SoftmaxRegression(lambda_=0.01)  # Specify the regularization strength
        sr.weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        sr.bias = np.array([[0.01, 0.01, 0.01]])
        X = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1, 0, 0], [0, 1, 0]])  # One-hot encoded

        # Compute the predicted probabilities
        scores = np.dot(X, sr.weights) + sr.bias
        y_pred = sr.softmax(scores)

        # Compute the expected loss manually, including L2 regularization
        epsilon = 1e-15  # To prevent log(0)
        log_probs = np.log(np.clip(y_pred, epsilon, 1 - epsilon))
        cross_entropy_loss = -np.mean(np.sum(y_true * log_probs, axis=1))
        l2_penalty = (sr.lambda_ / 2) * np.sum(np.square(sr.weights))
        expected_loss = cross_entropy_loss + l2_penalty

        # Compute the loss using the compute_loss method
        calculated_loss = sr.compute_loss(y_true, y_pred)

        self.assertAlmostEqual(expected_loss, calculated_loss, places=5)

    def test_gradient_descent_step(self):
        sr = SoftmaxRegression(learning_rate=0.01, n_iterations=10, lambda_=0.01)
        X = np.array([[1, 2], [1, 3], [1, 5], [1, 7]])
        y = np.array([0, 1, 2, 1])
        sr.weights = np.zeros((X.shape[1], len(np.unique(y))))
        sr.bias = np.zeros((1, len(np.unique(y))))
        y_one_hot = np.eye(len(np.unique(y)))[y]
        probabilities = sr.softmax(np.dot(X, sr.weights) + sr.bias)
        sr.gradient_descent_step(X, y_one_hot, probabilities)
        self.assertIsNotNone(sr.weights.any())
        self.assertIsNotNone(sr.bias.any())

    def test_fit(self):
        sr = SoftmaxRegression(learning_rate=0.01, n_iterations=1000, lambda_=0.01)
        X = np.array([[1, 2], [1, 3], [1, 5], [1, 7]])
        y = np.array([0, 1, 2, 1])
        sr.fit(X, y, print_loss=True)
        
        # Check if loss decreased
        self.assertLess(sr.loss_history[-1], sr.loss_history[0], "Loss did not decrease")

        # Check that at least one weight has been updated significantly from initialization
        # This checks for the maximum change in any single weight
        self.assertTrue(np.any(np.abs(sr.weights) > 1e-5), "No individual weight has been updated significantly")

        # Assuming bias is a numpy array and checking if it has changed significantly
        self.assertTrue(np.any(np.abs(sr.bias) > 1e-5), "Bias has not been updated significantly")

    def test_predict(self):
        sr = SoftmaxRegression(learning_rate=0.01, n_iterations=1000, lambda_=0.01)
        X = np.array([[1, 2], [1, 3], [1, 5], [1, 7]])
        y = np.array([0, 1, 2, 1])
        sr.fit(X, y)
        predictions = sr.predict(X)
        print("Predictions:", predictions)  # Print predictions
        print("True labels:", y)  # Print true labels
        # Check if the predicted labels are close to the true labels
        self.assertTrue(np.all(predictions == y) or np.sum(predictions == y) >= 3)

if __name__ == '__main__':
    unittest.main()