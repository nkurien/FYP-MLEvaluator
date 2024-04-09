from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from models.knn import KNearestNeighbours
from models.classification_tree import ClassificationTree
from models.logistic_regression import SoftmaxRegression
from models.optimisers import GridSearch
from data_processing.train_test_split import train_test_split

class TuningThread(QThread):
    tuning_completed = pyqtSignal(object, object, object, object, object, object)
    tuning_error = pyqtSignal(str)
    tuning_progress = pyqtSignal(int)  # Add the tuning_progress signal
    
    def __init__(self, X, y, preprocessor, model_params_grids=None, parent=None):
        super().__init__(parent)
        self.X = X
        self.y = y
        self.preprocessor = preprocessor
        if model_params_grids is not None:
            self.model_params_grids = model_params_grids # Use provided grids or default to empty dict
        else: self.model_params_grids = {}
        self.best_models = {}  # To store the best model from each tuning thread
        self.model_names = ['KNN', 'Classification Tree', 'Softmax Regression']
        self.knn_grid_search = None
        self.tree_grid_search = None
        self.softmax_grid_search = None
        self.tuning_aborted = False  # Flag to indicate if tuning is aborted

    
    def run(self):
        try:
            X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, seed=3456)
            
            X_train = self.preprocessor.fit_transform(X_train)
            X_val = self.preprocessor.transform(X_val)

            models = {
                'KNN': KNearestNeighbours(),
                'Classification Tree': ClassificationTree(),
                'Softmax Regression': SoftmaxRegression()
            }
            
            threads = []
            for model_name, model in models.items():
                if self.tuning_aborted:  # Check if tuning is aborted
                    break
                
                thread = ModelTuningThread(
                    model_name=model_name, 
                    model=model, 
                    X_train=X_train, 
                    y_train=y_train, 
                    X_val=X_val, 
                    y_val=y_val,  
                    grid_search_params=self.model_params_grids.get(model_name, None)
                )
                thread.tuning_completed.connect(self.on_tuning_completed)
                threads.append(thread)
                thread.start()
            
            total_threads = len(threads)
            completed_threads = 0

            for thread in threads:
                if self.tuning_aborted:  # Check if tuning is aborted
                    break
                
                thread.wait()  # Wait for the thread to complete
                completed_threads += 1
                progress = int((completed_threads / total_threads) * 100)
                self.tuning_progress.emit(progress)  # Emit the progress update

        except ValueError as e:
            self.tuning_error.emit(str(e))


    def on_tuning_completed(self, model_name, best_model, grid_search):
        if model_name == 'KNN':
            self.knn_grid_search = grid_search
        elif model_name == 'Classification Tree':
            self.tree_grid_search = grid_search
        elif model_name == 'Softmax Regression':
            self.softmax_grid_search = grid_search

        self.best_models[model_name] = best_model
        print(f"Tuning completed for {model_name}")

        if len(self.best_models) == len(self.model_names):
            # Emit the tuning_completed signal with the best models and GridSearch objects
            self.tuning_completed.emit(
                self.best_models.get('KNN'), self.best_models.get('Classification Tree'), self.best_models.get('Softmax Regression'),
                self.knn_grid_search, self.tree_grid_search, self.softmax_grid_search
            )

    def abort(self):
        self.tuning_aborted = True  # Set the abort flag


class ModelTuningThread(QThread):
    tuning_completed = pyqtSignal(str, object, object)  # Emits model name, best model, and GridSearch object

    def __init__(self, model_name, model, X_train, y_train, X_val, y_val, grid_search_params=None, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.model = model
        self.grid_search_params = grid_search_params  # Now optional
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.grid_search = None  # Add this line to store the GridSearch object

    
    def run(self):
        # Perform grid search
        self.grid_search = GridSearch(self.model, self.grid_search_params)
        self.grid_search.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        # Emit the best model and GridSearch object
        self.tuning_completed.emit(self.model_name, self.grid_search.best_model_, self.grid_search)
        print(f"Tuning completed for {self.model_name}")
