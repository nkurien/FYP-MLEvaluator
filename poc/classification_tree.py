import numpy as np
from train_test_split import train_test_split as split

class ClassificationTree :
    """
    A decision tree classifier that uses the Gini index for splitting.

    Attributes:
        max_depth (int): The maximum depth of the tree.
        min_size (int): The minimum number of samples required to split a node.
        root (dict): The root node of the decision tree.
    """
    def __init__(self, max_depth=20, min_size=1):
        """
        Constructs a ClassificationTree with specified maximum depth and minimum size.

        Args:
            max_depth (int): The maximum depth of the tree. Defaults to 20.
            min_size (int): The minimum number of samples required to split a node. Defaults to 1.
        """
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None

    def fit(self, X, y):
        """
        Fits the decision tree to the provided dataset.

        Args:
            X (array-like): Feature dataset.
            y (array-like): Target values.

        Combines X and y into a single dataset and builds the decision tree.
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Cannot fit a tree with an empty dataset")


        dataset = np.column_stack((X, y)).tolist()
        #print(dataset)
        self.root = self.build_tree(dataset, self.max_depth, self.min_size)

    def predict(self, X):
        """
        Predicts class labels for the given samples.

        Args:
            X (array-like): The input features to predict.

        Returns:
            np.array: Predicted class labels.
        """
        predictions = [self._predict(self.root, row) for row in X]
        return np.array(predictions)
    
    # Helper method for prediction that recursively traverses the tree to find the class for a single row.
    def _predict(self, node, row):
        """
        Helper method to predict the class for a single sample.

        Args:
            node (dict): The current node in the decision tree.
            row (list): A single sample from the dataset.

        Returns:
            The predicted class label for the sample.
        """
        if row[node['index']] < node['value']: # Check condition at the node.
            if isinstance(node['left'], dict): # If the left child is a dictionary, it's another decision node.
                return self._predict(node['left'], row)
            else:
                return node['left'] # If it's not a dictionary, it's a terminal node.
        else:
            if isinstance(node['right'], dict):
                return self._predict(node['right'], row)
            else:
                return node['right']

    # Main method that builds the tree from the training dataset.
    def build_tree(self, train, max_depth, min_size):
        """
        Builds the decision tree from the training dataset.

        Args:
            train (list): The training dataset.
            max_depth (int): The maximum depth of the tree.
            min_size (int): The minimum number of samples required to split a node.

        Returns:
            dict: The root node of the decision tree.
        """
        root = self.get_split(train) # Find the best initial split for the root.
        self.split(root, max_depth, min_size, 1) # Recursively split the tree from the root node.
        # The value 1 above is passed as the first layer depth from the root
        return root

    # Gini index function to evaluate the quality of a split.
    def gini_index(self, groups, classes):
        """
        Calculates the Gini index for a split.

        Args:
            groups (list): The groups of samples after a split.
            classes (list): The unique class values in the dataset.

        Returns:
            float: The Gini index for the split.
        """
        # Count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # Sum weighted Gini index for each group
        gini = 0.0 # Initialize Gini index.
        for group in groups:
            size = float(len(group))
            if size == 0:  # Avoid division by zero
                continue
            score = 0.0
            # Score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size # Proportion of each class.
                score += p * p
            # Sum weighted Gini index for each group.
            gini += (1.0 - score) * (size / n_instances)
        return gini
    
    # Method to find the best place to split the dataset.
    def get_split(self, dataset):
        """
        Finds the best split point in the dataset.

        Args:
            dataset (list): The dataset to split.

        Returns:
            dict: The best split point in the dataset.
        """
        class_values = list(set(row[-1] for row in dataset)) # Get the unique class values.
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1): # Iterate over all features
            for row in dataset:
                groups = self.test_split(index, row[index], dataset) # Test split on each unique feature value.
                gini = self.gini_index(groups, class_values) # Calculate Gini index for the split.
                if gini < b_score: # Check if we found a better split.
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups} # Return the best split.

    #Creates a terminal/leaf node from a group of samples. Called when splitting not necessary
    def to_terminal(self, group):
        """
        Creates a terminal node value.

        Args:
            group (list): The group of samples.

        Returns:
            The most common output value in the group (class label).
        """

        outcomes = [row[-1] for row in group] # Extract the outcomes from the group
        #This assumes the last column is the target variable
        #Returns the most frequent outcome
        return max(set(outcomes), key=outcomes.count)

    # Helper method to split the dataset based on an attribute and an attribute value
    # Called to help find all possible splits
    def test_split(self, index, value, dataset):
        """
        Splits a dataset based on an attribute and an attribute value.

        Args:
            index (int): The index of the attribute to split on.
            value: The value of the attribute to split on.
            dataset (list): The dataset to split.

        Returns:
            tuple: Two lists representing left and right splits of the dataset.
        """
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Recursive method to create child splits for a node or make terminal nodes.
    def split(self, node, max_depth, min_size, depth):
        """
        Recursively creates child splits for a node or makes terminal nodes.

        Args:
            node (dict): The node to split.
            max_depth (int): The maximum depth of the tree.
            min_size (int): The minimum number of samples required to split a node.
            depth (int): The current depth in the tree.

        This method splits nodes until the maximum depth or minimum node size is reached.
        """
        left, right = node['groups']
        del(node['groups'])
        # Check for no split
        if not left or not right :
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        
        # Check if we've reached maximum depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        
        # Process left child
        if len(left) <= min_size or self.is_homogeneous(left):
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth + 1)
        # Process right child
        if len(right) <= min_size or self.is_homogeneous(right):
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth + 1)
    
    # Prints the tree to the terminal in a relatively organised manner, left to right
    def print_tree(self, node=None, depth=0, feature_names=None):
        """
        Prints the decision tree to the output console. Each vertical line represents a layer.

        Args:
            node (dict, optional): The node to start printing from. Defaults to the root node.
            depth (int, optional): The initial depth for printing. Defaults to 0.
            feature_names (list of str, optional): Names of the features for better readability.

        This method recursively prints the tree from the given node.
        """
        if node is None:
            node = self.root
        
        # Print the current node condition
        if isinstance(node, dict):
            if feature_names:
                condition = f"{feature_names[node['index']]} <= {node['value']}"
            else:
                condition = f"X[{node['index']}] <= {node['value']}"
            print(f"{'|   ' * depth}{condition}")
            
            # Indicate and recursively print the left subtree
            print(f"{'|   ' * depth}Left:")
            if isinstance(node['left'], dict):
                self.print_tree(node['left'], depth + 1, feature_names)
            else:
                print(f"{'|   ' * (depth + 1)}--> Class: {node['left']}")
            
            # Indicate and recursively print the right subtree
            print(f"{'|   ' * depth}Right:")
            if isinstance(node['right'], dict):
                self.print_tree(node['right'], depth + 1, feature_names)
            else:
                print(f"{'|   ' * (depth + 1)}--> Class: {node['right']}")
        else:
            print(f"{'|   ' * depth}--> Class: {node}")
    
    def is_homogeneous(self, group):
        """
        Checks if all samples in the group belong to the same class.

        Args:
            group (list): The group of samples.

        Returns:
            bool: True if all samples in the group belong to the same class, False otherwise.
        """
        classes = [row[-1] for row in group]
        return len(set(classes)) == 1
    
    def get_depth(self, node=None):
        """
        Calculates the depth of the tree.

        Args:
            node (dict, optional): The node to calculate the depth from. Defaults to the root node.

        Returns:
            int: The depth of the tree.
        """
        if node is None:
            node = self.root

        # Check if the node is a leaf
        if not isinstance(node, dict):
            return 0  # Depth of a leaf node is 0

        # Recursively find the depth of the left and right subtrees
        left_depth = self.get_depth(node['left'])
        right_depth = self.get_depth(node['right'])

        # The depth of the node is the greater of the depths of its subtrees, plus one
        return max(left_depth, right_depth) + 1

