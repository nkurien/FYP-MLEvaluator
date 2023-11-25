import numpy as np
import train_test_split as split
from sklearn.datasets import load_iris

def k_folds(X, y, k=5, seed=None) :
    X, y = split.shuffle_data(X,y,seed)
    num_of_samples = len(y)
    # Check if 'k' is a valid value
    if k < 1 or k > num_of_samples:
        raise ValueError(f"'k' must be between 1 and {num_of_samples}, got {k}")
    indices = np.arange(num_of_samples)

    # Calculate the standard fold size
    fold_size = num_of_samples // k
    remainder = num_of_samples % k

    # Initialize an array to store the training and testing sets for each fold
    folds = []

    current = 0
    for i in range(k):
        if i < k - 1:
            start, stop = current, current + fold_size
        else:
            # For the last fold, use the remaining samples
            start, stop = current, num_of_samples

        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        folds.append(((X_train, y_train), (X_test, y_test)))

        current = stop

    return folds

iris = load_iris()
X = iris.data
y = iris.target
print(X.shape)
folds = k_folds(X,y, 10)

# Iterate over each fold
for i, fold in enumerate(folds):
    (X_train, y_train), (X_test, y_test) = fold

    print(f"Fold {i+1}:")
    print(f"  Training set: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    print(f"  Test set: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")
    print()  # Just to add an empty line for better readability
