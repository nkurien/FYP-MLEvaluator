import numpy as np
import train_test_split as split


def _k_folds(X, y, k=5, seed=None) :
    """
    Generates folds for a given dataset of features and labels (X, y)

    Returns:
        Folds as a list in the form of (X_train, y_train) (X_test, y_test)
        When iterating through, could be wise to retrieve each fold as shown:
            for i, fold in enumerate(folds):
            (X_train, y_train), (X_test, y_test) = fold


    """
    X, y = split.shuffle_data(X,y,seed)
    num_of_samples = len(y)
    # Check if 'k' is a valid value
    #K can't be 1, we accept 2 and higher
    if k < 2 or k > num_of_samples or not isinstance(k, int):
        raise ValueError(f"'k' must be an integer between 2 and {num_of_samples}, got {k}")
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


def k_folds_accuracy_scores(model, X, y, k=5, seed=None):
    """
    Returns a list of accuracy scores when given a model, dataset of features
    and labels, a number of folds (k) and a seed for shuffling the dataset 

    """


    # Generate folds using the previously defined k_folds function
    folds = _k_folds(X, y, k, seed)

    # List to store the scores from each fold
    scores = []

    # Iterate over each fold
    for (X_train, y_train), (X_test, y_test) in folds:
        # Fit the model on the training set
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Compute the score - here, we use accuracy as an example
        accuracy = np.mean(y_pred == y_test)

        # Append the score to the list
        scores.append(accuracy)

    return scores

def k_folds_accuracy_score(model, X, y, k=5, seed=None):
    """
    Returns the average accuracy score
    """
    return np.mean(k_folds_accuracy_scores(model, X, y, k, seed))


