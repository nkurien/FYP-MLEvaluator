import numpy as np
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from knn import KNearestNeighbours
from classification_tree import ClassificationTree
from logistic_regression import SoftmaxRegression


class GridSearch:
    def __init__(self, model, param_grid=None, scoring=None):
        """
        Initialize the GridSearch object.

        Parameters:
        -----------
        model : object
            The model object to perform grid search on.
        param_grid : dict
            A dictionary of parameter names and their corresponding values to search.
        scoring : callable or None, default=None
            The scoring function to evaluate the model's performance.
            If None, the model's default score method is used.
        """
        self.model = model
        if param_grid is None:
            self.param_grid = self._get_default_param_grid() 
        else:
            self.param_grid = param_grid
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        self.scores_ = []

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Fit the GridSearch object to the given training and validation data.

        Parameters:
        -----------
        X_train : array-like of shape (n_train_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_train_samples,)
            The target values for training.
        X_val : array-like of shape (n_val_samples, n_features)
            The validation input samples.
        y_val : array-like of shape (n_val_samples,)
            The target values for validation.
        """
        params_list = self._generate_params_list()
        scores = []

        for params in params_list:
            model = self._clone_model()
            model.__init__(**params)
            model.fit(X_train, y_train)
            score = self._validate(model, X_val, y_val)
            scores.append(score)

            if self.best_score_ is None or score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params
                self.best_model_ = model

        self.scores_ = scores  # Store the scores for each parameter combination

    def _clone_model(self):
        """
        Create a new instance of the model.

        Returns:
        --------
        model : object
            A new instance of the model.
        """
        return self.model.__class__()

    def _generate_params_list(self):
        """
        Generate a list of all possible parameter combinations from the parameter grid.

        Returns:
        --------
        params_list : list
            A list of dictionaries, where each dictionary represents a parameter combination.
        """
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        params_list = [dict(zip(param_names, params)) for params in product(*param_values)]
        return params_list
    
    def _get_default_param_grid(self):
        """
        Get the default parameter grid based on the model type.

        Returns:
        --------
        param_grid : dict
            The default parameter grid for the model.
        """
        if isinstance(self.model, KNearestNeighbours):
            param_grid = {
                'k': [1, 3, 5, 7, 9, 10, 20]
            }
        elif isinstance(self.model, ClassificationTree):
            param_grid = {
                'max_depth': [3, 5, 10, 15, 20],
                'min_size': [2, 5, 10]
            }
        elif isinstance(self.model, SoftmaxRegression):
            param_grid = {
                'learning_rate': [0.01, 0.1, 1],
                'n_iterations': [100, 500, 1000, 5000]
            }
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

        return param_grid

    def _validate(self, model, X_val, y_val):
        """
        Validate the model on the given validation data and return the score.

        Parameters:
        -----------
        model : object
            The model object to evaluate.
        X_val : array-like of shape (n_val_samples, n_features)
            The validation input samples.
        y_val : array-like of shape (n_val_samples,)
            The target values for validation.

        Returns:
        --------
        score : float
            The score of the model on the validation data.
        """
        if self.scoring is None:
            y_pred = model.predict(X_val)
            score = np.mean(y_pred == y_val)
        else:
            score = self.scoring(model, X_val, y_val)

        return score

    def score(self, X, y):
        """
        Returns the score of the best model on the given data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        --------
        score : float
            The score of the best model on the given data.
        """
        if self.scoring is None:
            y_pred = self.best_model_.predict(X)
            score = np.mean(y_pred == y)
        else:
            score = self.scoring(self.best_model_, X, y)

        return score

    def get_params(self, deep=True):
        """
        Get the parameters of the GridSearch object.

        Parameters:
        -----------
        deep : bool, default=True
            If True, return a deep copy of the parameters.

        Returns:
        --------
        params : dict
            The parameters of the GridSearch object.
        """
        return {
            'model': self.model,
            'param_grid': self.param_grid,
            'scoring': self.scoring
        }
    
    def get_scores(self):
     return self.scores_

    def set_params(self, **params):
        """
        Set the parameters of the GridSearch object.

        Parameters:
        -----------
        **params : dict
            The parameters to set.

        Returns:
        --------
        self : object
            The GridSearch object with updated parameters.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def display_param_grid(self):
        """
        Display the parameter grid.
        """
        print("Parameter Grid:")
        for param, values in self.param_grid.items():
            print(f"{param}: {values}")
    


    def plot_param_grid_heatmap(grid_search):
        """
        Plot a heatmap of the scores for each parameter combination in a grid search.

        Parameters:
        -----------
        grid_search : GridSearch object
            The GridSearch object containing the scores for each parameter combination.
        """
        # Get the parameter grid and scores
        param_grid = grid_search.param_grid
        scores = grid_search.scores_

        # Create a list of parameter combinations
        param_combinations = []
        for params in grid_search._generate_params_list():
            param_combinations.append(tuple(params.values()))

        # Create a 2D array of scores
        scores_array = np.array(scores).reshape(-1, 1)

        # Check if there is only one parameter
        if len(param_grid) == 1:
            # Plot a line chart for one-dimensional parameter grid
            plt.figure(figsize=(8, 6))
            param_name = list(param_grid.keys())[0]
            param_values = [pc[0] for pc in param_combinations]  # Extract parameter values
            plt.plot(param_values, scores, marker='o', linestyle='-')  # Plot as a line graph
            plt.title('Grid Search Scores')
            plt.xlabel(param_name)
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.grid(True) 
            plt.tight_layout()
        else:
            # Create a 2D array of parameter combinations
            param_combinations_array = np.array(param_combinations)

            # Get the unique values for each parameter
            param_names = list(param_grid.keys())
            param_values = [np.unique(param_combinations_array[:, i]) for i in range(len(param_names))]

            # Create a pivot table of scores
            scores_table = np.zeros((len(param_values[0]), len(param_values[1])))
            for i in range(len(param_combinations)):
                row_idx = np.where(param_values[0] == param_combinations[i][0])[0][0]
                col_idx = np.where(param_values[1] == param_combinations[i][1])[0][0]
                scores_table[row_idx, col_idx] = scores[i]
            
            print(scores_table)
            # Plot the heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(scores_table, annot=True, cmap='coolwarm', fmt='.3f',
                        xticklabels=param_values[1], yticklabels=param_values[0],
                        cbar_kws={'label': 'Score'})
            plt.title('Grid Search Scores')
            plt.xlabel(param_names[1])
            plt.ylabel(param_names[0])
            plt.tight_layout()

        plt.show()
