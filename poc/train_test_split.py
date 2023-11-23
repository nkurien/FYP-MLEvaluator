import numpy as np

'''Splits dataset into training and test sets '''

def shuffle_data(X, y, seed=None):
    '''Shuffles data samples'''
    if seed is not None:
        np.random.seed(seed)

    data_num = np.arange(X.shape[0])
    np.random.shuffle(data_num)

    return X[data_num], y[data_num]

def train_test_split(X, y, test_size=0.25, seed=None):  # Default test_size is now 0.25
    '''Splits dataset into training and test sets'''
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

