import numpy as np
from sklearn.datasets import load_iris
from train_test_split import train_test_split as split

class ClassificationTree :
    # Constructor to initialize the classification tree with max depth and minimum size for splitting.
    def __init__(self, max_depth=20, min_size=1):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None

    # The fit method combines the features and target variable to create a dataset and then builds the tree.
    def fit(self, X, y):
        # Combine X and y with column_stack and tolist to ease recursive calls during building
        # This is essentially our preprocessing for iris-data
        dataset = np.column_stack((X, y)).tolist()
        #print(dataset)
        self.root = self.build_tree(dataset, self.max_depth, self.min_size)

    # Predict method to make predictions for each sample in the feature matrix X.
    def predict(self, X):
        predictions = [self._predict(self.root, row) for row in X]
        return np.array(predictions)
    
    # Helper method for prediction that recursively traverses the tree to find the class for a single row.
    def _predict(self, node, row):
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
        root = self.get_split(train) # Find the best initial split for the root.
        self.split(root, max_depth, min_size, 1) # Recursively split the tree from the root node.
        # The value 1 above is passed as the first layer depth from the root
        return root

    # Gini index function to evaluate the quality of a split.
    def gini_index(self, groups, classes):
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
        outcomes = [row[-1] for row in group] # Extract the outcomes from the group
        #This assumes the last column is the target variable
        #Returns the most frequent outcome
        return max(set(outcomes), key=outcomes.count)

    # Helper method to split the dataset based on an attribute and an attribute value
    # Called to help find all possible splits
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Recursive method to create child splits for a node or make terminal nodes.
    def split(self, node, max_depth, min_size, depth):
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
        """Check if all samples in the tree node belong to the same class."""
        classes = [row[-1] for row in group]
        return len(set(classes)) == 1
    
    def get_depth(self, node=None):
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

