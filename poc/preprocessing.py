import numpy as np
import csv
import pandas as pd
from typing import List, Tuple, Union
from collections import Counter


class MinMaxScaler:
    """
    A scaler that scales features to a specified range.

    Attributes:
    -----------
    min_ : array-like of shape (n_features,)
        Per feature minimum seen in the data.
    max_ : array-like of shape (n_features,)
        Per feature maximum seen in the data.
    range_ : array-like of shape (n_features,)
        Per feature range (max - min) seen in the data.

    Methods:
    --------
    fit(X):
        Compute the minimum and maximum to be used for later scaling.
    transform(X):
        Scale features of X according to feature_range.
    fit_transform(X):
        Fit to data, then transform it.
    """
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, X):
        """
        Compute the minimum and maximum to be used for later scaling.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        Returns:
        --------
        self : object
            Fitted scaler.
        """
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.range_ = self.max_ - self.min_
        # Handle zero range to avoid division by zero
        self.range_[self.range_ == 0] = 1
        return self
        

    def transform(self, X):
        """
        Scale features of X according to feature_range.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.

        Returns:
        --------
        X_tr : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("The scaler has not been fitted yet.")
        return (X - self.min_) / self.range_

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.

        Returns:
        --------
        X_tr : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X).transform(X)

class SimpleImputer:
    """
    Imputation transformer for completing missing values.

    Attributes:
    -----------
    strategy : str, default='mean'
        The imputation strategy.
        - If "mean", then replace missing values using the mean along each column.
        - If "median", then replace missing values using the median along each column.
        - If "most_frequent", then replace missing using the most frequent value along each column.
        - If "constant", then replace missing values with fill_value.
    fill_value : str or numerical value, default=None
        When strategy == "constant", fill_value is used to replace all occurrences of missing_values.
        If left to the default, fill_value will be 0 when imputing numerical data.
    statistics_ : array-like of shape (n_features,)
        The calculated statistics used for imputation for each feature.

    Methods:
    --------
    fit(X):
        Calculate the statistics used for imputation.
    transform(X):
        Impute the missing values with computed statistics.
    fit_transform(X):
        Fit to data, then transform it.
    """
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X):
        """
        Calculate the statistics used for imputation.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data with missing values to be imputed.

        Returns:
        --------
        self : object
            Fitted imputer.
        """
        if isinstance(X, pd.DataFrame):
            # Convert DataFrame to numpy array
            X = X.values

        self.statistics_ = []
        for column in X.T:
            try:
                # Attempt to convert the column to float, ignoring empty spaces
                column_float = np.array([float(x) if x.strip() else np.nan for x in column])
                # Handle numeric columns
                if self.strategy == 'mean':
                    self.statistics_.append(np.nanmean(column_float))
                elif self.strategy == 'median':
                    self.statistics_.append(np.nanmedian(column_float))
                elif self.strategy == 'most_frequent':
                    self.statistics_.append(np.nanmax(np.bincount(column_float[~np.isnan(column_float)].astype(int))))
                elif self.strategy == 'constant':
                    self.statistics_.append(self.fill_value)
                else:
                    raise ValueError(f"Strategy '{self.strategy}' not supported for numeric data")
            except ValueError:
                # Handle non-numeric columns
                if self.strategy == 'most_frequent':
                    most_common = Counter(column[column != '']).most_common(1)
                    self.statistics_.append(most_common[0][0] if most_common else self.fill_value)
                elif self.strategy == 'constant':
                    self.statistics_.append(self.fill_value)
                else:
                    raise ValueError(f"Strategy '{self.strategy}' not supported for non-numeric data")

        self.statistics_ = np.array(self.statistics_)
        return self

    def transform(self, X):
        """
        Impute the missing values with computed statistics.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data with missing values to be imputed.

        Returns:
        --------
        X_tr : array-like of shape (n_samples, n_features)
            Transformed data with missing values imputed.
        """
        if self.statistics_ is None:
            raise RuntimeError("The imputer has not been fitted yet.")

        if isinstance(X, pd.DataFrame):
            # Convert DataFrame to numpy array
            X = X.values

        X_transformed = np.array(X, copy=True)
        for i, statistic in enumerate(self.statistics_):
            if np.issubdtype(X_transformed[:, i].dtype, np.number):
                # Handle numeric columns
                missing_idx = np.isnan(X_transformed[:, i])
            else:
                # Handle non-numeric columns
                missing_idx = (X_transformed[:, i] == '') | (X_transformed[:, i] == None)

            X_transformed[missing_idx, i] = statistic

        return X_transformed

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data with missing values to be imputed.

        Returns:
        --------
        X_tr : array-like of shape (n_samples, n_features)
            Transformed data with missing values imputed.
        """
        return self.fit(X).transform(X)

class OneHotEncoder:
    """
    One-hot encoder for encoding categorical features.

    Attributes:
    -----------
    categories_ : list of arrays
        The categories of each feature.

    Methods:
    --------
    fit(X):
        Learn the categories for each feature.
    transform(X):
        Transform X using one-hot encoding.
    fit_transform(X):
        Fit to data, then transform it.
    """
    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        """
        Learn the categories for each feature.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns:
        --------
        self : object
            Fitted encoder.
        """
        self.categories_ = [np.unique(col) for col in X.T]
        return self

    def transform(self, X):
        """
        Transform X using one-hot encoding.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to be transformed.

        Returns:
        --------
        X_tr : array-like of shape (n_samples, n_transformed_features)
            Transformed input.
        """
        outputs = []
        for i, categories in enumerate(self.categories_):
            # Create a zero matrix with shape (number of samples, number of categories)
            binary_matrix = np.zeros((X.shape[0], len(categories)))
            for j, category in enumerate(categories):
                binary_matrix[:, j] = (X[:, i] == category)
            outputs.append(binary_matrix)
        return np.hstack(outputs)

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to be transformed.

        Returns:
        --------
        X_tr : array-like of shape (n_samples, n_transformed_features)
            Transformed input.
        """
        return self.fit(X).transform(X)

class OrdinalEncoder:
    """
    Ordinal encoder for encoding categorical features as integers.

    Attributes:
    -----------
    categories_ : list of arrays
        The categories of each feature.
    ordinal_map_ : list of dicts
        The mapping from categories to integers for each feature.

    Methods:
    --------
    fit(X):
        Learn the categories and their ordinal mapping for each feature.
    transform(X):
        Transform X using ordinal encoding.
    fit_transform(X):
        Fit to data, then transform it.
    """
    def __init__(self):
        self.categories_ = None
        self.ordinal_map_ = None

    def fit(self, X):
        """
        Learn the categories and their ordinal mapping for each feature.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns:
        --------
        self : object
            Fitted encoder.
        """
        self.categories_ = [np.unique(col) for col in X.T]
        self.ordinal_map_ = []
        for categories in self.categories_:
            ordinal_map = {cat: i for i, cat in enumerate(categories)}
            self.ordinal_map_.append(ordinal_map)
        return self

    def transform(self, X):
        """
        Transform X using ordinal encoding.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to be transformed.

        Returns:
        --------
        X_tr : array-like of shape (n_samples, n_features)
            Transformed input.
        """
        X_encoded = np.zeros_like(X, dtype=int)
        for i, categories in enumerate(self.categories_):
            ordinal_map = self.ordinal_map_[i]
            X_encoded[:, i] = [ordinal_map.get(x, -1) for x in X[:, i]]  # Use -1 for unknown values
        return X_encoded

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to be transformed.

        Returns:
        --------
        X_tr : array-like of shape (n_samples, n_features)
            Transformed input.
        """
        return self.fit(X).transform(X)

class PreprocessingPipeline:
    """
    A pipeline for preprocessing data by applying a sequence of transformers.

    Parameters:
    -----------
    steps : list of (str, transformer) tuples
        List of (name, transform) tuples (implementing fit/transform) that are chained,
        in the order in which they are chained, with the last object an estimator.

    Attributes:
    -----------
    fitted_ : bool
        Flag to indicate if the pipeline has been fitted.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformers in the pipeline one after the other.
    transform(X):
        Transform the data, after the transformers are fitted.
    fit_transform(X, y=None):
        Fit the transformers and transform the data.
    """
    def __init__(self, steps: List[Tuple[str, object]]):
        self.steps = steps
        self._validate_steps()
        self.fitted_ = False  # Flag to check if the pipeline is fitted

    def _validate_steps(self):
        """
        Validate the steps in the pipeline.

        Raises:
        -------
        ValueError:
            If any step name is not a string.
        TypeError:
            If any transformer is missing required methods.
        """
        required_methods = ['fit', 'transform']
        for name, transformer in self.steps:
            if not isinstance(name, str):
                raise ValueError(f"Step name '{name}' is not a string.")
            if not all(hasattr(transformer, method) for method in required_methods):
                missing_methods = [method for method in required_methods if not hasattr(transformer, method)]
                raise TypeError(f"Transformer '{type(transformer).__name__}' is missing required methods: {', '.join(missing_methods)}.")

    def fit(self, X, y=None):
        """
        Fit the transformers in the pipeline one after the other.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        y : None
            Not used, present for API consistency by convention.

        Returns:
        --------
        self : object
            Fitted pipeline.
        """
        for _, transformer in self.steps:
            if hasattr(transformer, 'fit_transform'):
                X = transformer.fit_transform(X)  # Only for steps that directly depend on y
            else:
                transformer.fit(X, y)
                X = transformer.transform(X)  # Apply transform separately
        self.fitted_ = True  # Mark the pipeline as fitted
        return self

    def transform(self, X):
        """
        Transform the data, after the transformers are fitted.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to transform, where n_samples is the number of samples and n_features is the number of features.

        Returns:
        --------
        X_tr : array-like of shape (n_samples, n_transformed_features)
            Transformed data.

        Raises:
        -------
        RuntimeError:
            If the pipeline has not been fitted.
        """
        if not self.fitted_:
            raise RuntimeError("The pipeline has not been fitted yet. Call 'fit' before 'transform'.")
        for _, transformer in self.steps:
            X = transformer.transform(X)
        return X

    def fit_transform(self, X, y=None):
        """
        Fit the transformers and transform the data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        y : None
            Not used, present for API consistency by convention.

        Returns:
        --------
        X_tr : array-like of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        return self.fit(X, y).transform(X)
    

class LabelEncoder:
    """
    Label encoder for encoding labels as integers.

    Attributes:
    -----------
    classes_ : array-like of shape (n_classes,)
        Holds the label for each class.
    mapping_ : dict
        Mapping from original labels to encoded labels.
    inverse_mapping_ : dict
        Mapping from encoded labels to original labels.
    unseen_label : int, default=-1
        Value to use for unseen labels during transform.

    Methods:
    --------
    fit(y):
        Fit label encoder to labels.
    transform(y):
        Transform labels to normalized encoding.
    inverse_transform(y):
        Transform labels back to original encoding.
    fit_transform(y):
        Fit label encoder and return encoded labels.
    """
    def __init__(self):
        self.classes_ = None
        self.mapping_ = None
        self.inverse_mapping_ = None
        self.unseen_label = -1  # Default value for unseen labels

    def fit(self, y):
        """Fit label encoder to labels."""
        # Check for higher-dimensional inputs
        if y.ndim > 2:
            raise ValueError("LabelEncoder expects input with 1 or 2 dimensions, got {}.".format(y.ndim))
        # Flatten y to 1D if it's 2D (shape: [n_samples, 1])
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        self.classes_ = np.unique(y)
        self.mapping_ = {label: idx for idx, label in enumerate(self.classes_)}
        self.inverse_mapping_ = {idx: label for label, idx in self.mapping_.items()}
        return self

    def transform(self, y):
        """Transform labels to normalized encoding."""
        # Check and flatten as in fit
        if y.ndim > 2:
            raise ValueError("LabelEncoder expects input with 1 or 2 dimensions, got {}.".format(y.ndim))
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        return np.array([self.mapping_.get(label, -1) for label in y])  # Assuming unseen_label is -1

    def inverse_transform(self, y):
        """Transform labels back to original encoding."""
        # Check as in fit and transform
        if isinstance(y, list):
            y = np.array(y)
        if y.ndim > 2:
            raise ValueError("LabelEncoder expects input with 1 or 2 dimensions, got {}.".format(y.ndim))
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        return np.array([self.inverse_mapping_.get(label) for label in y])

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels."""
        return self.fit(y).transform(y)

class NumericConverter:
    """
    Converter for transforming non-numeric data to numeric.

    Methods:
    --------
    fit(X, y=None):
        Fit the converter to the data (no-op).
    transform(X):
        Transform non-numeric data to numeric.
    fit_transform(X, y=None):
        Fit to data, then transform it.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_numeric = np.empty_like(X, dtype=float)
        for i in range(X.shape[1]):
            try:
                X_numeric[:, i] = X[:, i].astype(float)
            except ValueError:
                X_numeric[:, i] = np.nan
        return X_numeric

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class CombinedPreprocessor:
    """
    A preprocessor that combines multiple preprocessing pipelines for different subsets of features.

    Parameters:
    -----------
    **pipelines : dict
        A dictionary of preprocessing pipelines and their corresponding column indices.
        Each key-value pair should be in the format: 'pipeline_name': (pipeline, columns),
        where 'pipeline_name' is a string representing the name of the pipeline,
        'pipeline' is an instance of a preprocessing pipeline, and 'columns' is a list
        of column indices to which the pipeline should be applied. If 'columns' is None,
        the pipeline will be applied to the entire dataset.

    Methods:
    --------
    fit(X, y=None):
        Fit the preprocessing pipelines to the input data.
    transform(X):
        Transform the input data using the fitted preprocessing pipelines.
    fit_transform(X, y=None):
        Fit the preprocessing pipelines to the input data and transform the data.
    """

    def __init__(self, **pipelines):
        """
        Initialize the CombinedPreprocessor.

        Parameters:
        -----------
        **pipelines : dict
            A dictionary of preprocessing pipelines and their corresponding column indices.
        """
        self.pipelines = pipelines

    def fit(self, X, y=None):
        """
        Fit the preprocessing pipelines to the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data to be preprocessed.
        y : None
            Ignored. This parameter exists only for compatibility with sklearn transformers.

        Returns:
        --------
        self : object
            The fitted CombinedPreprocessor.

        Raises:
        -------
        ValueError
            If an error occurs during fitting any of the preprocessing pipelines.
        """
        for pipeline_name, (pipeline, columns) in self.pipelines.items():
            try:
                if columns is not None:
                    X_subset = X[:, columns]
                    pipeline.fit(X_subset)
                else:
                    pipeline.fit(X)
            except ValueError as e:
                raise ValueError(f"Error fitting {pipeline_name} pipeline: {str(e)}")
        return self

    def transform(self, X):
        """
        Transform the input data using the fitted preprocessing pipelines.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data to be preprocessed.

        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_transformed_features)
            The transformed data after applying the preprocessing pipelines.

        Raises:
        -------
        ValueError
            If an error occurs during transforming any of the preprocessing pipelines
            or during stacking the transformed subsets.
        """
        transformed_subsets = []
        for pipeline_name, (pipeline, columns) in self.pipelines.items():
            try:
                if columns is not None:
                    X_subset = X[:, columns]
                    transformed_subset = pipeline.transform(X_subset)
                else:
                    transformed_subset = pipeline.transform(X)
                transformed_subsets.append(transformed_subset)
            except ValueError as e:
                raise ValueError(f"Error transforming {pipeline_name} pipeline: {str(e)}")
        try:
            X_transformed = np.hstack(transformed_subsets)
            return X_transformed
        except ValueError as e:
            raise ValueError(f"Error stacking transformed subsets: {str(e)}")

    def fit_transform(self, X, y=None):
        """
        Fit the preprocessing pipelines to the input data and transform the data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data to be preprocessed.
        y : None
            Ignored. This parameter exists only for compatibility with sklearn transformers.

        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_transformed_features)
            The transformed data after fitting and applying the preprocessing pipelines.
        """
        self.fit(X, y)
        return self.transform(X)

def load_dataset(file_path, target_col=-1, sep=',', missing_values=None, drop_missing=False, dtype=None, header=False):
    """
    Load a dataset from a file and split it into features (X) and target (y) using NumPy.

    Parameters:
    -----------
    file_path : str
        The path to the dataset file.
    target_col : int, default=-1
        The index of the target column. Default is -1 (last column).
    sep : str, default=','
        The separator used in the dataset file. Default is ','. Use '\t' for tab-separated files and ' ' for space-separated files.
    missing_values : list or None, default=None
        The values to consider as missing. Default is None.
    drop_missing : bool, default=False
        Whether to drop rows with missing values. Default is False.
    dtype : numpy.dtype or None, default=None
        The data type of the loaded data. Default is None.
    header : bool, default=False
        Whether the dataset has a header row. Default is False.

    Returns:
    --------
    X : array-like of shape (n_samples, n_features)
        The feature matrix.
    y : array-like of shape (n_samples,)
        The target vector.

    Raises:
    -------
    FileNotFoundError
        If the specified file is not found.
    IOError
        If an error occurs while reading the file.
    ValueError
        If the file format or target column index is invalid.
    Exception
        If an error occurs while loading the dataset.
    """
    try:
        data = []
        with open(file_path, 'r') as file:
            if sep == '\t' :
                for line in file:
                    row = line.strip().split()
                    data.append(row)
            else:
                csv_reader = csv.reader(file, delimiter=sep)
                if header:
                    next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    data.append([value.strip() for value in row])

        data = np.array(data, dtype=dtype)

        # Handle missing values
        if missing_values:
            mask = np.isin(data, missing_values)
            data[mask] = np.nan
        if drop_missing:
            data = data[~np.isnan(data).any(axis=1)]

        # Extract the features (X) and target (y)
        if target_col < 0:
            target_col = data.shape[1] + target_col
        if 0 <= target_col < data.shape[1]:
            if target_col == data.shape[1] - 1:
                X = data[:, :-1]
            else:
                X = np.concatenate((data[:, :target_col], data[:, target_col+1:]), axis=1)
            y = data[:, target_col]
        else:
            raise ValueError(f"Invalid target column index: {target_col}")

        return X, y

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except IOError as e:
        raise IOError(f"An error occurred while reading the file: {e}")
    except ValueError as ve:
        raise ValueError(f"Invalid file format or target column: {ve}")
    except Exception as e:
        raise Exception(f"An error occurred while loading the dataset: {e}")