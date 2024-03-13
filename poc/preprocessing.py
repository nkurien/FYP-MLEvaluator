import numpy as np
import csv
import pandas as pd
from typing import List, Tuple, Union
from collections import Counter


class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, X):
        """Fit the scaler to the data."""
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.range_ = self.max_ - self.min_
        # Handle zero range to avoid division by zero
        self.range_[self.range_ == 0] = 1
        return self
        

    def transform(self, X):
        """Transform the data using the fitted scaler."""
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("The scaler has not been fitted yet.")
        return (X - self.min_) / self.range_

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

class SimpleImputer:
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X):
        """Calculate the fill value depending on the strategy."""
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
        """Fill in missing values in X."""
        if self.statistics_ is None:
            raise RuntimeError("The imputer has not been fitted yet.")

        if isinstance(X, pd.DataFrame):
            # Convert DataFrame to numpy array
            X = X.values

        X_transformed = np.array(X, copy=True, dtype=object)
        for i, statistic in enumerate(self.statistics_):
            try:
                # Attempt to convert the column to float
                column_float = X_transformed[:, i].astype(float)
                # Handle numeric columns
                missing_idx = np.isnan(column_float)
            except ValueError:
                # Handle non-numeric columns
                missing_idx = (X_transformed[:, i] == '') | (X_transformed[:, i] == None)

            X_transformed[missing_idx, i] = statistic

        return X_transformed

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def transform(self, X):
        """Fill in missing values in X."""
        if self.statistics_ is None:
            raise RuntimeError("The imputer has not been fitted yet.")

        if isinstance(X, pd.DataFrame):
            # Convert DataFrame to numpy array
            X = X.values

        X_transformed = np.array(X, copy=True, dtype=object)
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
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def transform(self, X):
        """Fill in missing values in X."""
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
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
    

class OneHotEncoder:
    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        """Learn the categories for each feature."""
        self.categories_ = [np.unique(col) for col in X.T]
        return self

    def transform(self, X):
        """Transform X using one-hot encoding."""
        outputs = []
        for i, categories in enumerate(self.categories_):
            # Create a zero matrix with shape (number of samples, number of categories)
            binary_matrix = np.zeros((X.shape[0], len(categories)))
            for j, category in enumerate(categories):
                binary_matrix[:, j] = (X[:, i] == category)
            outputs.append(binary_matrix)
        return np.hstack(outputs)

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
    
    import numpy as np

class OrdinalEncoder:
    def __init__(self):
        self.categories_ = None
        self.ordinal_map_ = None

    def fit(self, X):
        """Learn the categories and their ordinal mapping for each feature."""
        self.categories_ = [np.unique(col) for col in X.T]
        self.ordinal_map_ = []
        for categories in self.categories_:
            ordinal_map = {cat: i for i, cat in enumerate(categories)}
            self.ordinal_map_.append(ordinal_map)
        return self

    def transform(self, X):
        """Transform X using ordinal encoding."""
        X_encoded = np.zeros_like(X, dtype=int)
        for i, categories in enumerate(self.categories_):
            ordinal_map = self.ordinal_map_[i]
            X_encoded[:, i] = [ordinal_map.get(x, -1) for x in X[:, i]]  # Use -1 for unknown values
        return X_encoded

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

class PreprocessingPipeline:
    def __init__(self, steps: List[Tuple[str, object]]):
        self.steps = steps
        self._validate_steps()
        self.fitted_ = False  # Flag to check if the pipeline is fitted

    def _validate_steps(self):
        required_methods = ['fit', 'transform']
        for name, transformer in self.steps:
            if not isinstance(name, str):
                raise ValueError(f"Step name '{name}' is not a string.")
            if not all(hasattr(transformer, method) for method in required_methods):
                missing_methods = [method for method in required_methods if not hasattr(transformer, method)]
                raise TypeError(f"Transformer '{type(transformer).__name__}' is missing required methods: {', '.join(missing_methods)}.")

    def fit(self, X, y=None):
        for _, transformer in self.steps:
            if hasattr(transformer, 'fit_transform'):
                X = transformer.fit_transform(X)  # Only for steps that directly depend on y
            else:
                transformer.fit(X, y)
                X = transformer.transform(X)  # Apply transform separately
        self.fitted_ = True  # Mark the pipeline as fitted
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("The pipeline has not been fitted yet. Call 'fit' before 'transform'.")
        for _, transformer in self.steps:
            X = transformer.transform(X)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    

class LabelEncoder:
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
    def __init__(self, categorical_pipeline, numerical_pipeline, ordinal_pipeline, cat_columns, num_columns, ord_columns):
        self.categorical_pipeline = categorical_pipeline
        self.numerical_pipeline = numerical_pipeline
        self.ordinal_pipeline = ordinal_pipeline
        self.categorical_columns = cat_columns
        self.numerical_columns = num_columns
        self.ordinal_columns = ord_columns

    def fit(self, X, y=None):
        X_categorical = X[:, self.categorical_columns]
        X_numerical = X[:, self.numerical_columns]
        X_ordinal = X[:, self.ordinal_columns]

        self.categorical_pipeline.fit(X_categorical)
        self.numerical_pipeline.fit(X_numerical)
        self.ordinal_pipeline.fit(X_ordinal)

        return self

    def transform(self, X):
        X_categorical = X[:, self.categorical_columns]
        X_numerical = X[:, self.numerical_columns]
        X_ordinal = X[:, self.ordinal_columns]

        X_categorical_transformed = self.categorical_pipeline.transform(X_categorical)
        X_numerical_transformed = self.numerical_pipeline.transform(X_numerical)
        X_ordinal_transformed = self.ordinal_pipeline.transform(X_ordinal)

        X_transformed = np.hstack((X_categorical_transformed, X_numerical_transformed, X_ordinal_transformed))

        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)






import csv
import numpy as np

def load_dataset(file_path, target_col=-1, sep=',', missing_values=None, drop_missing=False, dtype=None, header=False):
    """
    Load a dataset from a file and split it into features (X) and target (y) using NumPy.

    Args:
        file_path (str): The path to the dataset file.
        target_col (int): The index of the target column. Default is -1 (last column).
        sep (str): The separator used in the dataset file. Default is ','. Use '\t' for tab-separated files and ' ' for space-separated files.
        missing_values (list or None): The values to consider as missing. Default is None.
        drop_missing (bool): Whether to drop rows with missing values. Default is False.
        dtype (numpy.dtype or None): The data type of the loaded data. Default is None.
        header (bool): Whether the dataset has a header row. Default is False.

    Returns:
        tuple: A tuple containing the features (X) and target (y) as numpy arrays.
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