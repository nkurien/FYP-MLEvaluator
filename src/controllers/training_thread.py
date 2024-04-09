from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from data_processing.cross_validation import *
from data_processing.metrics import *

class TrainingThread(QThread):
    model_progress = pyqtSignal(int)
    model_scores = pyqtSignal(str, list)
    training_completed = pyqtSignal(list, list, list, list, dict, dict, dict)  # Add dictionaries for metrics
    
    def __init__(self, X, y, k, seed, preprocessor, best_knn, best_tree, best_softmax, parent=None):
        super().__init__(parent)
        self.X = X
        self.y = y
        self.k = k
        self.seed = seed
        self.preprocessor = preprocessor
        self.best_knn = best_knn
        self.best_tree = best_tree
        self.best_softmax = best_softmax
        self.training_aborted = False
    
    def run(self):
        # Create instances of the models
        knn_metrics = {}
        tree_metrics = {}
        softmax_metrics = {}
        models = [
            ('KNN', self.best_knn),
            ('Classification Tree', self.best_tree),
            ('Softmax Regression', self.best_softmax)
        ]
        
        # Create the preprocessor within the training thread
       # preprocessor = PreprocessingPipeline([
       #     ("imputer", SimpleImputer(strategy="mean")),
       #    ("converter", NumericConverter()),
       #     ("scaler", MinMaxScaler())
       # ])
        
        labels = np.unique(self.y).tolist()  # Convert labels to a list
        num_models = len(models)
        y_preds = []
        
        for i, (model_name, model) in enumerate(models):
            if self.training_aborted:
                break
            
            # Perform cross-validation and get accuracy scores
            scores = k_folds_accuracy_scores(model, self.X, self.y, self.k, self.seed, self.preprocessor)
            self.model_scores.emit(model_name, scores)
            
            # Get the predicted labels for confusion matrix
            y_true, y_pred = k_folds_predictions(model, self.X, self.y, self.k, self.seed, self.preprocessor)
            y_preds.append(y_pred)
            
            #Calculate and save metrics from k-folds matrix
            if model_name == 'KNN':
                knn_metrics = calculate_metrics(y_true, y_pred)
            elif model_name == 'Classification Tree':
                tree_metrics = calculate_metrics(y_true, y_pred)
            elif model_name == 'Softmax Regression':
                softmax_metrics = calculate_metrics(y_true, y_pred)

            # Update progress
            progress = int((i + 1) / num_models * 100)
            self.model_progress.emit(progress)
        
        if not self.training_aborted:
            self.training_completed.emit(scores, y_true, y_preds, labels, knn_metrics, tree_metrics, softmax_metrics)

    
    def abort(self):
        self.training_aborted = True