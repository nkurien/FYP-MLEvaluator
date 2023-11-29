import unittest
import numpy as np
import sys
sys.path.append("..")
from classification_tree import ClassificationTree  # Assuming the class is in classification_tree.py

class TestClassificationTree(unittest.TestCase):

    def setUp(self):
        # Set up a simple dataset for testing
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y = np.array([0, 1, 0, 1])


    def test_initialization(self):
        """Test the initialization of the ClassificationTree."""
        tree = ClassificationTree(max_depth=10, min_size=2)
        self.assertEqual(tree.max_depth, 10)
        self.assertEqual(tree.min_size, 2)

    def test_fit(self):
        """Test fitting the model with a dataset."""
        tree = ClassificationTree()
        tree.fit(self.X, self.y)
        self.assertIsNotNone(tree.root)

    def test_predict(self):
        """Test making predictions."""
        tree = ClassificationTree()
        tree.fit(self.X, self.y)
        predictions = tree.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def test_gini_index(self):
        """Test the Gini index calculation."""
        tree = ClassificationTree()
        groups = [[[1, 0], [2, 0]], [[3, 1], [4, 1]]]
        classes = [0, 1]
        gini = tree.gini_index(groups, classes)
        self.assertGreaterEqual(gini, 0)
        self.assertLessEqual(gini, 1)
    
    def test_tree_depth(self):
        """Test if the tree respects the maximum depth limit."""
        tree = ClassificationTree(max_depth=3)
        tree.fit(self.X, self.y)
        depth = tree.get_depth()
        self.assertLessEqual(depth, 3)

    def test_homogeneity_check(self):
        """Test if the is_homogeneous function correctly identifies a homogeneous group."""
        tree = ClassificationTree()
        homogeneous_group = [[1, 0], [2, 0]]
        heterogeneous_group = [[1, 0], [2, 1]]
        self.assertTrue(tree.is_homogeneous(homogeneous_group))
        self.assertFalse(tree.is_homogeneous(heterogeneous_group))


    def test_terminal_node_creation(self):
        """Test if terminal nodes are correctly created from a group of samples."""
        tree = ClassificationTree()
        group = [[0,0], [1, 0], [2, 0], [3, 1], [4, 1]]
        terminal_node = tree.to_terminal(group)
        # The most common class in the group is 0
        self.assertEqual(terminal_node, 0)

    def test_splitting_functionality(self):
        """Test if the dataset is correctly split based on a feature and value."""
        tree = ClassificationTree()
        dataset = [[1, 0], [2, 1], [3, 0], [4, 1]]
        left, right = tree.test_split(0, 2.5, dataset)
        # Expecting left split to have first two rows and right split to have last two rows
        self.assertEqual(left, [[1, 0], [2, 1]])
        self.assertEqual(right, [[3, 0], [4, 1]])

    def test_tree_structure(self):
        """Test the structure of the tree by checking the properties of the root node."""
        tree = ClassificationTree()
        tree.fit(self.X, self.y)
        self.assertIsInstance(tree.root, dict)
        self.assertIn('index', tree.root)
        self.assertIn('value', tree.root)
        self.assertIn('left', tree.root)
        self.assertIn('right', tree.root)

    def test_empty_dataset(self):
        """Test how the tree handles an empty dataset."""
        tree = ClassificationTree()
        with self.assertRaises(ValueError):
            tree.fit(np.array([]), np.array([]))


if __name__ == '__main__':
    unittest.main()
