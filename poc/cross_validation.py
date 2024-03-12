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
    #unused but may be useful if we optimise remainder-handling

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


def k_folds_accuracy_scores(model, X, y, k=5, seed=None, preprocessor=None,):
    """
    Returns a list of accuracy scores when given a model, dataset of features
    and labels, a preprocessor (optional), a number of folds (k), and a seed for shuffling the dataset.

    """
    # Check if the model has 'fit' and 'predict' methods
    if not hasattr(model, 'fit') or not callable(getattr(model, 'fit')):
        raise AttributeError("The provided model does not have a callable 'fit' method.")
    if not hasattr(model, 'predict') or not callable(getattr(model, 'predict')):
        raise AttributeError("The provided model does not have a callable 'predict' method.")

    # Check if the preprocessor has 'fit' and 'transform' methods (if provided)
    if preprocessor is not None:
        if not hasattr(preprocessor, 'fit') or not callable(getattr(preprocessor, 'fit')):
            raise AttributeError("The provided preprocessor does not have a callable 'fit' method.")
        if not hasattr(preprocessor, 'transform') or not callable(getattr(preprocessor, 'transform')):
            raise AttributeError("The provided preprocessor does not have a callable 'transform' method.")

    # Generate folds using the previously defined k_folds function
    folds = _k_folds(X, y, k, seed)

    # List to store the scores from each fold
    scores = []

    # Iterate over each fold
    for (X_train, y_train), (X_test, y_test) in folds:
        # Preprocess the training and testing data (if preprocessor is provided)
        if preprocessor is not None:
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

        # Fit the model on the preprocessed training set
        model.fit(X_train, y_train)

        # Predict on the preprocessed test set
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

def k_folds_predictions(model, X, y, k=5, seed=None):
    """
    Perform k-fold cross-validation to generate predictions for each data point in the dataset.

    This function splits the dataset into k folds, trains the provided model on k-1 of those folds, 
    and then predicts the remaining fold. This process is repeated for each fold, resulting in 
    predictions for each data point in the dataset. The true labels and predictions are aggregated 
    and returned.

    Parameters:
    - model: A machine learning model instance that implements fit and predict methods.
    - X: ndarray, shape (n_samples, n_features)
      The input data to be split and used for training and testing.
    - y: ndarray, shape (n_samples,)
      The target labels corresponding to the input data.
    - k: int, default=5
      The number of folds to split the data into for cross-validation.
    - seed: int or None, optional
      The seed for the random number generator used to shuffle the data before splitting into folds.
      If None, the random number generator is the RandomState instance used by np.random.

    Returns:
    - all_true_labels: list
      The list of true labels for each data point, aggregated across all k folds.
    - all_predictions: list
      The list of predictions for each data point, aggregated across all k folds.

    """
    folds = _k_folds(X, y, k, seed)
    all_predictions = []
    all_true_labels = []

    for (X_train, y_train), (X_test, y_test) in folds:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        all_predictions.extend(predictions)
        all_true_labels.extend(y_test)
    
    return all_true_labels, all_predictions


def leave_one_out_scores(model, X, y, seed=None):
    k = len(y)

    return k_folds_accuracy_scores(model, X, y, k, seed)

def leave_one_out_score(model, X, y, seed=None):
    return np.mean(leave_one_out_scores(model,X,y,seed))


