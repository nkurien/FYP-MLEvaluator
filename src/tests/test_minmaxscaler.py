import unittest
import numpy as np
import sys
sys.path.append("..")
from sklearn.datasets import load_iris
from data_processing.train_test_split import train_test_split
from data_processing.preprocessing import MinMaxScaler

class TestMinMaxScaler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load and split the Iris dataset
        iris = load_iris()
        iono = np.genfromtxt("../datasets/ionosphere.txt", delimiter=',')
        cls.iris_X = iris.data
        cls.iris_y = iris.target
        cls.iris_X_train, cls.iris_X_test, cls.iris_y_train, cls.iris_y_test = train_test_split(cls.iris_X, cls.iris_y, test_size=0.2, seed=42)

        cls.iono_X = iono[:, :-1]  # All columns except the last one
        cls.iono_y = iono[:, -1]   # Last column
        cls.iono_X_train, cls.iono_X_test, cls.iono_y_train, cls.iono_y_test = train_test_split(cls.iono_X, cls.iono_y, test_size=0.2, seed=42)

    def test_iris_scaling(self):
        scaler = MinMaxScaler()
        scaler.fit(self.iris_X_train)
        scaled_X_train = scaler.transform(self.iris_X_train)
        scaled_X_test = scaler.transform(self.iris_X_test)

        # Test if training data is scaled properly
        self.assertTrue(np.all(scaled_X_train >= 0) and np.all(scaled_X_train <= 1))

        # Test if test data is scaled properly
        self.assertTrue(np.all(scaled_X_test >= 0) and np.all(scaled_X_test <= 1))

    def test_ionosphere_scaling(self):
        scaler = MinMaxScaler()
        scaler.fit(self.iono_X_train)
        scaled_X_train = scaler.transform(self.iono_X_train)
        scaled_X_test = scaler.transform(self.iono_X_test)

        # Test if training data is scaled properly
        self.assertTrue(np.all(scaled_X_train >= 0) and np.all(scaled_X_train <= 1))

        # Test if test data is scaled properly
        self.assertTrue(np.all(scaled_X_test >= 0) and np.all(scaled_X_test <= 1))






if __name__ == '__main__':
    unittest.main()