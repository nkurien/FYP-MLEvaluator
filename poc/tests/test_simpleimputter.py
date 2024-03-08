import unittest
import numpy as np
import sys
sys.path.append("..")
from preprocessing import SimpleImputer

class TestSimpleImputer(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset with missing values
        self.X = np.array([[1, 2, np.nan], [4, np.nan, 6], [np.nan, 8, 9]])

    

if __name__ == '__main__':
    unittest.main()