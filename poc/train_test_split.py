import numpy as np

'''Splits dataset into training and test sets '''

def shuffle_data(X, y, seed=None):
    """
    Shuffles the data samples along with their corresponding labels.

    Args:
        X (array-like): Data features, a 2D array of shape (n_samples, n_features).
        y (array-like): Data labels, a 1D array of shape (n_samples,).
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple: Shuffled data features and labels.
    """
    if seed is not None:
        np.random.seed(seed)

    data_num = np.arange(X.shape[0])
    np.random.shuffle(data_num)

    return X[data_num], y[data_num]

def train_test_split(X, y, test_size=0.25, seed=None):  # Default test_size is now 0.25
    """
    Splits the dataset into training and test sets.

    Args:
        X (array-like): Data features, a 2D array of shape (n_samples, n_features).
        y (array-like): Data labels, a 1D array of shape (n_samples,).
        test_size (float or int, optional): If float, represents the proportion of 
            the dataset to include in the test split. If int, represents the 
            absolute number of test samples. Defaults to 0.25.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple: Split data into training and test sets (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If test_size is not in the range 0 < test_size < 1 or not an int less than the number of samples.
    """
    # Shuffle is now done by default
    X, y = shuffle_data(X, y, seed)

    num_samples = len(y)
    
    if 0 <= test_size < 1:
        train_ratio = num_samples - int(num_samples * test_size)
    elif 1 <= test_size < num_samples:
        train_ratio = num_samples - test_size
    else:
        raise ValueError("Invalid test_size value")

    X_train, X_test = X[:train_ratio], X[train_ratio:]
    y_train, y_test = y[:train_ratio], y[train_ratio:]

    return X_train, X_test, y_train, y_test

