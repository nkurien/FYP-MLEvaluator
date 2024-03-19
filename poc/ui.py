import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QCheckBox, QScrollArea,QProgressBar, QInputDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from preprocessing import load_dataset
from knn import KNearestNeighbours
from classification_tree import ClassificationTree
from logistic_regression import SoftmaxRegression
from cross_validation import k_folds_accuracy_scores, k_folds_predictions
from preprocessing import PreprocessingPipeline, CombinedPreprocessor, SimpleImputer, MinMaxScaler, NumericConverter, OrdinalEncoder, OneHotEncoder
from train_test_split import train_test_split
from optimisers import GridSearch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

class ConfusionMatrixPlot(FigureCanvas):
    def __init__(self, parent=None, width=15, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.axes = None  # Initialize axes later based on the number of matrices

    # Use Matplotlib version 3.7.3! s
    def plot_confusion_matrices(self, y_true, y_preds, labels, titles):
        # Clear existing axes if they exist
        if self.axes is not None:
            for ax in self.axes:
                ax.clear()  # Clear each subplot
            # Remove existing axes from the figure
            self.figure.clf()
        num_matrices = len(y_preds)
        self.axes = self.figure.subplots(1, num_matrices)  # Create subplots based on the number of predictions

        if num_matrices == 1:
            self.axes = [self.axes]  # Wrap it in a list if only one axes object

        colours = ['Blues', 'Oranges', 'Greens', 'Reds']

        for i, (y_pred, title) in enumerate(zip(y_preds, titles)):
            cm = confusion_matrix(y_true, y_pred)
            print(cm)
            sns.heatmap(cm, annot=True, cmap=colours[i%3], xticklabels=labels, yticklabels=labels,
                        square=True, cbar=False, fmt='d', ax=self.axes[i])
            self.axes[i].set_xlabel('Predicted')
            self.axes[i].set_ylabel('True')
            self.axes[i].set_title(title)

        self.figure.tight_layout()  # Adjust layout to fit everything
        self.draw()  # Redraw the canvas with the new content




class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Dataset Loader')
        self.setGeometry(100, 100, 1250, 900)  
        self.init_ui()
    
    def init_ui(self):
        self.main_layout = QVBoxLayout()

        # Scroll Area Setup
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)  # Important for the scroll area to adapt
        self.scroll_widget = QWidget()  # This widget will hold the main layout
        self.scroll_widget.setLayout(self.main_layout)
        self.scroll_area.setWidget(self.scroll_widget)

        # Setup layout for the entire window
        layout = QVBoxLayout(self)  # This layout is for the QMainWindow itself
        layout.addWidget(self.scroll_area)  # Add the scroll area to the main window layout
        self.progress_bar = QProgressBar() #Progress bar for training

        # File selection layout
        file_layout = QHBoxLayout()
        self.file_label = QLabel('Dataset File:')
        self.file_text = QLineEdit()
        self.file_button = QPushButton('Browse')
        self.file_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_text)
        file_layout.addWidget(self.file_button)

        # Parameters layout
        params_layout = QHBoxLayout()
        self.target_col_label = QLabel('Target Column Index:')
        self.target_col_text = QLineEdit('-1')  # Default value
        self.target_col_text.setFixedWidth(50)  
        self.delimiter_label = QLabel('Delimiter:') 
        self.delimiter_text = QLineEdit(',')  # Default value
        self.delimiter_text.setFixedWidth(50) 
        self.header_checkbox = QCheckBox("Remove Header")
        params_layout.addWidget(self.target_col_label)
        params_layout.addWidget(self.target_col_text)
        params_layout.addWidget(self.delimiter_label)
        params_layout.addWidget(self.delimiter_text)
        params_layout.addWidget(self.header_checkbox)  #Checkbox for header removal

        self.load_button = QPushButton('Load Dataset')
        self.load_button.clicked.connect(self.load_dataset)

        self.preview_label = QLabel('')
        self.preview_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.preview_label.setWordWrap(True)

        self.train_button = QPushButton('Train Models')
        self.train_button.setEnabled(False)
        self.train_button.clicked.connect(self.train_models)
        self.main_layout.addWidget(self.progress_bar)

        self.process_button = QPushButton('Process Data')
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.process_data)

        self.column_types_layout = QHBoxLayout()
    
        self.categorical_label = QLabel('Categorical Columns:')
        self.categorical_text = QLineEdit()
        self.categorical_text.setPlaceholderText('Enter comma-separated indices')
        
        self.numerical_label = QLabel('Numerical Columns:')
        self.numerical_text = QLineEdit()
        self.numerical_text.setPlaceholderText('Enter comma-separated indices')
        
        self.ordinal_label = QLabel('Ordinal Columns:')
        self.ordinal_text = QLineEdit()
        self.ordinal_text.setPlaceholderText('Enter comma-separated indices')
        
        self.column_types_layout.addWidget(self.categorical_label)
        self.column_types_layout.addWidget(self.categorical_text)
        self.column_types_layout.addWidget(self.numerical_label)
        self.column_types_layout.addWidget(self.numerical_text)
        self.column_types_layout.addWidget(self.ordinal_label)
        self.column_types_layout.addWidget(self.ordinal_text)

        self.categorical_data_label = QLabel('')
        self.numerical_data_label = QLabel('')
        self.ordinal_data_label = QLabel('')  

        self.tune_button = QPushButton('Tune Models')
        self.tune_button.setEnabled(False)
        self.tune_button.clicked.connect(self.tune_models)

        self.best_knn_label = QLabel('')
        self.best_tree_label = QLabel('')
        self.best_softmax_label = QLabel('')
        
        self.tuning_plot_widget = TuningPlotWidget(self)


        self.abort_button = QPushButton('Abort')
        self.abort_button.setVisible(False)
        self.abort_button.clicked.connect(self.abort_training)


        self.knn_label = QLabel('KNN Accuracy Scores:')
        self.tree_label = QLabel('Classification Tree Accuracy Scores:')
        self.softmax_label = QLabel('Softmax Regression Accuracy Scores:')

        self.confusion_matrix_plot = ConfusionMatrixPlot(self, width=15, height=4, dpi=100)


        # Adding widgets to the main layout
       # Adding widgets to the main_layout, which is set to the scroll_widget
        self.main_layout.addLayout(file_layout)
        self.main_layout.addLayout(params_layout)
        self.main_layout.addWidget(self.load_button)
        self.main_layout.addWidget(self.preview_label)
        self.main_layout.addLayout(self.column_types_layout)
        self.main_layout.addWidget(self.process_button)
        self.main_layout.addWidget(self.categorical_data_label)
        self.main_layout.addWidget(self.numerical_data_label)
        self.main_layout.addWidget(self.ordinal_data_label)
        self.main_layout.addWidget(self.tune_button)
        self.main_layout.addWidget(self.best_knn_label)
        self.main_layout.addWidget(self.best_tree_label)
        self.main_layout.addWidget(self.best_softmax_label)
        self.main_layout.addWidget(self.tuning_plot_widget)
        self.main_layout.addWidget(self.train_button)
        self.main_layout.addWidget(self.abort_button)
        self.main_layout.addWidget(self.knn_label)
        self.main_layout.addWidget(self.tree_label)
        self.main_layout.addWidget(self.softmax_label)
        self.main_layout.addWidget(self.confusion_matrix_plot)


        self.setLayout(layout)
    
    def browse_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Dataset File', '', 'All Files (*)', options=options)
        if file_path:
            self.file_text.setText(file_path)
    
    def load_dataset(self):
        file_path = self.file_text.text()
        header = self.header_checkbox.isChecked()
        
        try:
            target_col = int(self.target_col_text.text())
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Target column index must be an integer.')
            return
        
        delimiter = self.delimiter_text.text()
        # Handle special case for tab delimiter
        if delimiter.lower() == "tab" or delimiter == "\\t":
            delimiter = '\t'  # Convert to actual tab character
        
        if file_path:
            try:
                self.X, self.y = load_dataset(file_path, target_col=target_col, sep=delimiter, header=header)
                QMessageBox.information(self, 'Dataset Loaded', 'Dataset has been loaded successfully.')
                self.train_button.setEnabled(True)
                self.process_button.setEnabled(True)

                # Display the first five lines of the loaded dataset
                preview_text = 'Dataset Preview:\n'
                for i in range(min(5, len(self.X))):
                    preview_text += f'Line {i+1}: {self.X[i]}\n'
                self.preview_label.setText(preview_text)
                preview_text += f'Labels: \n'
                for i in range(min(5, len(self.X))):
                    preview_text += f'{i+1}: {self.y[i]} \n'
                
                # Display the shape of X
                preview_text += f'\nShape of X: {self.X.shape}'

                self.preview_label.setText(preview_text)
            except Exception as e:
                QMessageBox.critical(self, 'Error Loading Dataset', str(e))
        else:
            QMessageBox.warning(self, 'No File Selected', 'Please select a dataset file.')
    
    def process_data(self):
        try:
            # Retrieve column indices from the text boxes
            categorical_columns = self.parse_column_indices(self.categorical_text.text())
            numerical_columns = self.parse_column_indices(self.numerical_text.text())
            ordinal_columns = self.parse_column_indices(self.ordinal_text.text())

            # Check for 'all' input and set columns accordingly
            if categorical_columns == ['all']:
                categorical_columns = list(range(self.X.shape[1]))
            if numerical_columns == ['all']:
                numerical_columns = list(range(self.X.shape[1]))
            if ordinal_columns == ['all']:
                ordinal_columns = list(range(self.X.shape[1]))

            # Check for duplicate column indices
            all_columns = categorical_columns + numerical_columns + ordinal_columns
            if len(set(all_columns)) != len(all_columns):
                raise ValueError("Duplicate column indices found.")

            # Check if column indices are within valid range
            max_column_index = self.X.shape[1] - 1
            if any(col > max_column_index for col in all_columns):
                raise ValueError(f"Column index out of range. Maximum column index is {max_column_index}.")

            # Create the preprocessor based on the column types
            self.preprocessor = self.create_preprocessor(categorical_columns, numerical_columns, ordinal_columns)

            # Display the data points for each column type
            if len(categorical_columns) > 0:
                self.categorical_data_label.setText(f"Categorical columns: {self.X[0][categorical_columns]}")
            else:
                self.categorical_data_label.setText("Categorical columns: None")

            if len(numerical_columns) > 0:
                self.numerical_data_label.setText(f"Numerical columns: {self.X[0][numerical_columns]}")
            else:
                self.numerical_data_label.setText("Numerical columns: None")

            if len(ordinal_columns) > 0:
                self.ordinal_data_label.setText(f"Ordinal columns: {self.X[0][ordinal_columns]}")
            else:
                self.ordinal_data_label.setText("Ordinal columns: None")

            QMessageBox.information(self, 'Data Processed', 'Data has been processed successfully.')
            self.tune_button.setEnabled(True)  # Enable the "Tune Models" button

        except ValueError as e:
            QMessageBox.warning(self, 'Invalid Input', str(e))

    def parse_column_indices(self, text):
        if text.strip().lower() == 'all':
            return ['all']
        else:
            try:
                return list(map(int, text.split(','))) if text.strip() else []
            except ValueError:
                raise ValueError("Invalid column indices. Please enter valid integers separated by commas.")
    

    def create_preprocessor(self, categorical_columns, numerical_columns, ordinal_columns):
        # Create the individual preprocessing pipelines
        categorical_pipeline = PreprocessingPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder())
        ])

        numerical_pipeline = PreprocessingPipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("converter", NumericConverter()),
            ("scaler", MinMaxScaler())
        ])

        ordinal_pipeline = PreprocessingPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder())
        ])

        # Create the combined preprocessor
        preprocessor = CombinedPreprocessor(
            num = (numerical_pipeline, numerical_columns),
            cat = (categorical_pipeline, categorical_columns),
            ord = (ordinal_pipeline, ordinal_columns)
        )

        return preprocessor
    
    
    def tune_models(self):
        # Preparation of data and preprocessor should already be done
        self.tuning_thread = TuningThread(self.X, self.y, self.preprocessor)
        self.tuning_thread.tuning_completed.connect(self.on_tuning_completed)
        self.tuning_thread.start()
        self.tune_button.setEnabled(False)  # Disable the buttons while tuning is in progress
        self.train_button.setEnabled(False)
    
    def on_tuning_completed(self, best_knn, best_tree, best_softmax, knn_grid_search, tree_grid_search, softmax_grid_search):
        self.best_knn = best_knn
        self.best_tree = best_tree
        self.best_softmax = best_softmax

        # Display the best model parameters
        self.best_knn_label.setText(f"KNN: K={self.best_knn.k}")
        self.best_tree_label.setText(f"Classification Tree: Max Depth={self.best_tree.max_depth}, Min Size={self.best_tree.min_size}")
        self.best_softmax_label.setText(f"Softmax Regression: Learning Rate={self.best_softmax.learning_rate}, N Iterations={self.best_softmax.n_iterations}")

        # Plot the tuning results
        self.tuning_plot_widget.plot_tuning_results(knn_grid_search, tree_grid_search, softmax_grid_search)

        QMessageBox.information(self, 'Tuning Complete', 'Models have been tuned successfully.')
        self.train_button.setEnabled(True)  # Enable the "Train Models" button

    
    def abort_training(self):
        self.training_thread.abort()
        self.abort_button.setVisible(False)
        self.train_button.setEnabled(True)
        QMessageBox.information(self, 'Training Aborted', 'Training has been aborted.')

    
    def train_models(self):
        self.training_thread = TrainingThread(self.X, self.y, k=5, seed=2108, preprocessor=self.preprocessor,
                                            best_knn=self.best_knn, best_tree=self.best_tree, best_softmax=self.best_softmax)
        self.training_thread.model_progress.connect(self.update_progress)
        self.training_thread.model_scores.connect(self.display_scores)
        self.training_thread.training_completed.connect(self.on_training_completed)
        self.training_thread.start()
        
        self.abort_button.setVisible(True)
        self.train_button.setEnabled(False)
        self.progress_bar.setValue(0)
    
    def update_progress(self, progress):
        self.progress_bar.setValue(progress)
    
    def display_scores(self, model_name, scores):
        if model_name == 'KNN':
            self.knn_label.setText(f'KNN Accuracy Scores:\n{scores}')
        elif model_name == 'Classification Tree':
            self.tree_label.setText(f'Classification Tree Accuracy Scores:\n{scores}')
        elif model_name == 'Softmax Regression':
            self.softmax_label.setText(f'Softmax Regression Accuracy Scores:\n{scores}')
    
    def plot_confusion_matrix(self, y_true, y_pred, labels, model_name):
        self.confusion_matrix_plot.plot_confusion_matrices(y_true, [y_pred], labels, [model_name])
    
    def on_training_completed(self, scores, y_true, y_preds, labels):
        self.confusion_matrix_plot.plot_confusion_matrices(y_true, y_preds, labels, ['KNN', 'Classification Tree', 'Softmax Regression'])
        QMessageBox.information(self, 'Training Complete', 'Models have been trained successfully.')
        self.abort_button.setVisible(False)
        self.train_button.setEnabled(True)
    
class TrainingThread(QThread):
    model_progress = pyqtSignal(int)
    model_scores = pyqtSignal(str, list)
    training_completed = pyqtSignal(list, list, list, list)
    
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
            
            # Update progress
            progress = int((i + 1) / num_models * 100)
            self.model_progress.emit(progress)
        
        if not self.training_aborted:
            self.training_completed.emit(scores, y_true, y_preds, labels)
    
    def abort(self):
        self.training_aborted = True

class TuningThread(QThread):
    tuning_completed = pyqtSignal(object, object, object, object, object, object)

    
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
    
    def run(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, seed=2108)
        
        X_train = self.preprocessor.fit_transform(X_train)
        X_val = self.preprocessor.transform(X_val)

        models = {
            'KNN': KNearestNeighbours(),
            'Classification Tree': ClassificationTree(),
            'Softmax Regression': SoftmaxRegression()
        }
        
        threads = []
        for model_name, model in models.items():
            thread = ModelTuningThread(
                model_name=model_name, 
                model=model, 
                X_train=X_train, 
                y_train=y_train, 
                X_val=X_val, 
                y_val=y_val,  
                grid_search_params=self.model_params_grids.get(model_name, None)  # Safely get params or None
            )
            thread.tuning_completed.connect(self.on_tuning_completed)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
           thread.wait()  # Wait for all threads to complete

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

class TuningPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(10, 3), dpi=100)  # Adjust the figure size here
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

    def plot_tuning_results(self, knn_grid_search, tree_grid_search, softmax_grid_search):
        self.figure.clear()
        
        # Create subplots with adjusted width and height ratios
        gs = self.figure.add_gridspec(1, 3, width_ratios=[1, 1, 1], height_ratios=[1])
        ax1 = self.figure.add_subplot(gs[0])
        ax2 = self.figure.add_subplot(gs[1])
        ax3 = self.figure.add_subplot(gs[2])

        # Plot KNN line graph
        ax1.plot(knn_grid_search.param_grid['k'], knn_grid_search.scores_, marker='o', linestyle='-')
        ax1.set_title('KNN Tuning Results')
        ax1.set_xlabel('K')
        ax1.set_ylabel('Score')
        ax1.grid(True)

        # Plot Classification Tree heatmap
        tree_scores_table = self.reshape_scores(tree_grid_search.scores_, tree_grid_search.param_grid)
        sns.heatmap(tree_scores_table, annot=True, cmap='coolwarm', fmt='.3f',
                    xticklabels=tree_grid_search.param_grid['min_size'],
                    yticklabels=tree_grid_search.param_grid['max_depth'],
                    ax=ax2, cbar_kws={'label': 'Score'})
        ax2.set_title('Classification Tree Tuning Results')
        ax2.set_xlabel('Min Size')
        ax2.set_ylabel('Max Depth')

        # Plot Softmax Regression heatmap
        softmax_scores_table = self.reshape_scores(softmax_grid_search.scores_, softmax_grid_search.param_grid)
        sns.heatmap(softmax_scores_table, annot=True, cmap='coolwarm', fmt='.3f',
                    xticklabels=softmax_grid_search.param_grid['n_iterations'],
                    yticklabels=softmax_grid_search.param_grid['learning_rate'],
                    ax=ax3, cbar_kws={'label': 'Score'})
        ax3.set_title('Softmax Regression Tuning Results')
        ax3.set_xlabel('N Iterations')
        ax3.set_ylabel('Learning Rate')

        self.figure.tight_layout()
        self.canvas.draw()

    def reshape_scores(self, scores, param_grid):
        # Reshape scores into a 2D array based on parameter grid
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        scores_table = np.reshape(scores, (len(param_values[0]), len(param_values[1])))
        return scores_table

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

