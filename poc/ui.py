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
        self.setGeometry(100, 100, 800, 600)  
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
        # Retrieve column indices from the text boxes
        categorical_columns = self.parse_column_indices(self.categorical_text.text())
        numerical_columns = self.parse_column_indices(self.numerical_text.text())
        ordinal_columns = self.parse_column_indices(self.ordinal_text.text())
        
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

    def parse_column_indices(self, text):
        return list(map(int, text.split(','))) if text.strip() else []
    
    def prompt_column_types(self):
        # Create a dialog or input fields for the user to enter column types
        # You can use QInputDialog or create a custom dialog
        # For simplicity, let's assume the user enters comma-separated indices for each type
        categorical_input, ok = QInputDialog.getText(self, 'Categorical Columns', 'Enter comma-separated indices of categorical columns:')
        numerical_input, ok = QInputDialog.getText(self, 'Numerical Columns', 'Enter comma-separated indices of numerical columns:')
        ordinal_input, ok = QInputDialog.getText(self, 'Ordinal Columns', 'Enter comma-separated indices of ordinal columns:')

        categorical_columns = list(map(int, categorical_input.split(','))) if categorical_input else []
        numerical_columns = list(map(int, numerical_input.split(','))) if numerical_input else []
        ordinal_columns = list(map(int, ordinal_input.split(','))) if ordinal_input else []

        return categorical_columns, numerical_columns, ordinal_columns

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
        self.tuning_thread = TuningThread(self.X, self.y, self.preprocessor)
        self.tuning_thread.tuning_completed.connect(self.on_tuning_completed)
        self.tuning_thread.start()
        
        self.tune_button.setEnabled(False)
        self.train_button.setEnabled(False)
    
    def on_tuning_completed(self, best_knn, best_tree, best_softmax):
        self.best_knn = best_knn
        self.best_tree = best_tree
        self.best_softmax = best_softmax
        
        # Display the best model parameters
        self.best_knn_label.setText(f"KNN: K={best_knn.k}")
        self.best_tree_label.setText(f"Classification Tree: Max Depth={best_tree.max_depth}, Min Size={best_tree.min_size}")
        self.best_softmax_label.setText(f"Softmax Regression: Learning Rate={best_softmax.learning_rate}, N Iterations={best_softmax.n_iterations}")

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
    tuning_completed = pyqtSignal(object, object, object)
    
    def __init__(self, X, y, preprocessor, parent=None):
        super().__init__(parent)
        self.X = X
        self.y = y
        self.preprocessor = preprocessor
    
    def run(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, seed=2108)
        
        # Create instances of the models
        knn_model = KNearestNeighbours()
        tree_model = ClassificationTree()
        softmax_model = SoftmaxRegression()
        
        # Create GridSearch objects for each model
        knn_grid_search = GridSearch(knn_model)
        tree_grid_search = GridSearch(tree_model)
        softmax_grid_search = GridSearch(softmax_model)
        
        X_train = self.preprocessor.fit_transform(X_train)
        X_val = self.preprocessor.transform(X_val)


        # Perform grid search for each model
        knn_grid_search.fit(X_train, y_train, X_val, y_val)
        tree_grid_search.fit(X_train, y_train, X_val, y_val)
        softmax_grid_search.fit(X_train, y_train, X_val, y_val)
        
        # Get the best models
        best_knn = knn_grid_search.best_model_
        best_tree = tree_grid_search.best_model_
        best_softmax = softmax_grid_search.best_model_
        
        self.tuning_completed.emit(best_knn, best_tree, best_softmax)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

