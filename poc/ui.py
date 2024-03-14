import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QCheckBox, QScrollArea, QSizePolicy,QProgressBar
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5 import QtGui
from preprocessing import load_dataset
from knn import KNearestNeighbours
from classification_tree import ClassificationTree
from logistic_regression import SoftmaxRegression
from cross_validation import k_folds_accuracy_scores, k_folds_predictions
from preprocessing import PreprocessingPipeline, CombinedPreprocessor, SimpleImputer, MinMaxScaler, NumericConverter
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
        self.scroll_widget = QWidget()  # This widget will hold your main layout
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

                # Display the first five lines of the loaded dataset
                preview_text = 'Dataset Preview:\n'
                for i in range(min(5, len(self.X))):
                    preview_text += f'Line {i+1}: {self.X[i]}\n'
                self.preview_label.setText(preview_text)
                preview_text += f'Labels: \n'
                for i in range(min(5, len(self.X))):
                    preview_text += f'{i+1}: {self.y[i]} \n'
                self.preview_label.setText(preview_text)
            except Exception as e:
                QMessageBox.critical(self, 'Error Loading Dataset', str(e))
        else:
            QMessageBox.warning(self, 'No File Selected', 'Please select a dataset file.')
    
    def abort_training(self):
        self.training_thread.abort()
        self.abort_button.setVisible(False)
        self.train_button.setEnabled(True)
        QMessageBox.information(self, 'Training Aborted', 'Training has been aborted.')

    
    def train_models(self):
        self.training_thread = TrainingThread(self.X, self.y, k=5, seed=42)
        self.training_thread.training_completed.connect(self.on_training_completed)
        self.training_thread.start()
        
        self.abort_button.setVisible(True)
        self.train_button.setEnabled(False)
    
    def on_training_completed(self, scores_list, y_true, y_preds, labels):
        # Display the accuracy scores
        self.knn_label.setText(f'KNN Accuracy Scores:\n{scores_list[0]}')
        self.tree_label.setText(f'Classification Tree Accuracy Scores:\n{scores_list[1]}')
        self.softmax_label.setText(f'Softmax Regression Accuracy Scores:\n{scores_list[2]}')
        
        # Plot the confusion matrices
        self.confusion_matrix_plot.plot_confusion_matrices(y_true, y_preds, labels, ['KNN', 'Classification Tree', 'Softmax Regression'])
        
        QMessageBox.information(self, 'Training Complete', 'Models have been trained successfully.')
        
        self.abort_button.setVisible(False)
        self.train_button.setEnabled(True)
    
class TrainingThread(QThread):
    training_completed = pyqtSignal(list, list, list, list)
    
    def __init__(self, X, y, k, seed, parent=None):
        super().__init__(parent)
        self.X = X
        self.y = y
        self.k = k
        self.seed = seed
        self.training_aborted = False
    
    def run(self):
        # Create instances of the models
        models = [
            ('KNN', KNearestNeighbours(k=5)),
            ('Classification Tree', ClassificationTree(max_depth=5)),
            ('Softmax Regression', SoftmaxRegression(learning_rate=0.1, n_iterations=1000))
        ]
        
        # Create the preprocessor within the training thread
        preprocessor = PreprocessingPipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("converter", NumericConverter()),
            ("scaler", MinMaxScaler())
        ])
        
        labels = np.unique(self.y).tolist()  # Convert labels to a list
        y_preds = []
        scores_list = []
        
        for model_name, model in models:
            if self.training_aborted:
                break
            
            # Perform cross-validation and get accuracy scores
            scores = k_folds_accuracy_scores(model, self.X, self.y, self.k, self.seed, preprocessor)
            scores_list.append(scores)
            
            # Get the predicted labels for confusion matrix
            y_true, y_pred = k_folds_predictions(model, self.X, self.y, self.k, self.seed, preprocessor)
            y_preds.append(y_pred)
        
        if not self.training_aborted:
            self.training_completed.emit(scores_list, y_true, y_preds, labels)
    
    def abort(self):
        self.training_aborted = True

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

