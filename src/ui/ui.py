import sys
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QCheckBox, QScrollArea,QProgressBar, QDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from data_processing.preprocessing import load_dataset
from controllers.training_thread import TrainingThread
from controllers.tuning_thread import TuningThread
from controllers.preprocessing_handler import PreprocessingHandler
from ui.confusion_matrix import ConfusionMatrixPlot
from ui.tuning_widget import TuningPlotWidget
from ui.advanced_settings import AdvancedSettingsDialog
import numpy as np

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Model Evaluator')
        self.setGeometry(100, 100, 1250, 900) 
        self.setWindowIcon(QIcon('resources/icon.png'))
        self.init_ui()

        self.num_folds = 5  # Default value
        self.training_seed = 2108 #Default
        
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

        self.train_button = QPushButton('Evaluate Models')
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

        self.metrics_layout = QHBoxLayout()
        
        self.knn_metrics_label = QLabel('KNN Metrics:')
        self.knn_metrics_label.setAlignment(Qt.AlignLeft)
        self.knn_metrics_label.setVisible(False)  # Set initial visibility to False
        self.knn_metrics_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.metrics_layout.addWidget(self.knn_metrics_label)
        
        self.tree_metrics_label = QLabel('Classification Tree Metrics:')
        self.tree_metrics_label.setAlignment(Qt.AlignCenter)
        self.tree_metrics_label.setVisible(False)  # Set initial visibility to False
        self.tree_metrics_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.metrics_layout.addWidget(self.tree_metrics_label)
        
        self.softmax_metrics_label = QLabel('Softmax Regression Metrics:')
        self.softmax_metrics_label.setAlignment(Qt.AlignRight)
        self.softmax_metrics_label.setVisible(False)  # Set initial visibility to False
        self.softmax_metrics_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.metrics_layout.addWidget(self.softmax_metrics_label)

        self.advanced_settings_button = QPushButton("Evaluation Settings")
        self.advanced_settings_button.clicked.connect(self.open_advanced_settings) 


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
        self.main_layout.addWidget(self.advanced_settings_button)
        self.main_layout.addWidget(self.knn_label)
        self.main_layout.addWidget(self.tree_label)
        self.main_layout.addWidget(self.softmax_label)
        self.main_layout.addWidget(self.confusion_matrix_plot)
        self.main_layout.addLayout(self.metrics_layout)


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
        handler = PreprocessingHandler()
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
            handler.set_data(self.X)
            handler.set_column_indices(categorical_columns, numerical_columns, ordinal_columns)
            self.preprocessor = handler.create_preprocessor()

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
    
    def tune_models(self):
        try:
            # Preparation of data and preprocessor should already be done
            self.tuning_thread = TuningThread(self.X, self.y, self.preprocessor)
            self.tuning_thread.tuning_completed.connect(self.on_tuning_completed)
            self.tuning_thread.tuning_progress.connect(self.update_progress)  # Connect progress signal
            self.tuning_thread.tuning_error.connect(self.on_tuning_error)
            self.tuning_thread.start()
            print("Tuning started")
            self.tune_button.setEnabled(False)  # Disable the buttons while tuning is in progress
            self.train_button.setEnabled(False)
            self.abort_button.setVisible(True)  # Show the abort button
            self.abort_button.clicked.disconnect()  # Disconnect previous connection
            self.abort_button.clicked.connect(self.abort_tuning)  # Connect abort button to tuning abort
        except ValueError as e:
            QMessageBox.warning(self, 'Tuning Error', str(e) + '\nPlease check if the columns were inputted correctly during the preprocessing phase.')


    
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
        self.abort_button.setVisible(False)  # Hide the abort button
        self.progress_bar.setValue(0)  # Reset progress bar to 0%
    
    def on_tuning_error(self, error_message):
        QMessageBox.warning(self, 'Tuning Error', error_message + '\nPlease check if the columns were inputted correctly during the preprocessing phase.')
    
    def abort_tuning(self):
        self.tuning_thread.abort()
        self.abort_button.setVisible(False)
        self.tune_button.setEnabled(True)
        QMessageBox.information(self, 'Tuning Aborted', 'Tuning has been aborted.')
        print("Tuning Aborted")
        self.progress_bar.setValue(0)  # Reset progress bar to 0%
    
    def abort_training(self):
        self.training_thread.abort()
        self.abort_button.setVisible(False)
        self.train_button.setEnabled(True)
        QMessageBox.information(self, 'Training Aborted', 'Training has been aborted.')

    
    def train_models(self):
        self.training_thread = TrainingThread(self.X, self.y, k=self.num_folds, seed=self.training_seed, preprocessor=self.preprocessor,
                                            best_knn=self.best_knn, best_tree=self.best_tree, best_softmax=self.best_softmax)
        self.training_thread.model_progress.connect(self.update_progress)
        self.training_thread.model_scores.connect(self.display_scores)
        self.training_thread.training_completed.connect(self.on_training_completed)
        self.training_thread.start()
        
        self.abort_button.setVisible(True)
        self.abort_button.clicked.disconnect()  # Disconnect previous connection
        self.abort_button.clicked.connect(self.abort_training)  # Connect abort button to training abort
        self.train_button.setEnabled(False)
        self.progress_bar.setValue(0)
    
    def update_progress(self, progress):
        self.progress_bar.setValue(progress)
    
    def open_advanced_settings(self):
        current_settings = {
            "num_folds": self.num_folds,
            "seed": self.training_seed,  # Add the current seed value
        }
        dialog = AdvancedSettingsDialog(current_settings, self)
        if dialog.exec_() == QDialog.Accepted:
            settings = dialog.get_settings()
            self.update_advanced_settings(settings)

    def update_advanced_settings(self, settings):
        # Update the corresponding variables in MainWindow
        self.num_folds = settings["num_folds"]
        self.training_seed = settings["seed"]  # Update the seed value
    
    def display_scores(self, model_name, scores):
        if model_name == 'KNN':
            self.knn_scores = scores
            self.knn_label.setText(f'KNN Accuracy Scores:\n{scores}')
        elif model_name == 'Classification Tree':
            self.tree_scores = scores
            self.tree_label.setText(f'Classification Tree Accuracy Scores:\n{scores}')
        elif model_name == 'Softmax Regression':
            self.softmax_scores = scores
            self.softmax_label.setText(f'Softmax Regression Accuracy Scores:\n{scores}')
    
    def plot_confusion_matrix(self, y_true, y_pred, labels, model_name):
        self.confusion_matrix_plot.plot_confusion_matrices(y_true, [y_pred], labels, [model_name])
    
    def on_training_completed(self, scores, y_true, y_preds, labels, knn_metrics, tree_metrics, softmax_metrics):
        self.confusion_matrix_plot.plot_confusion_matrices(y_true, y_preds, labels, ['KNN', 'Classification Tree', 'Softmax Regression'])
        QMessageBox.information(self, 'Training Complete', 'Models have been trained successfully.')
        
        knn_mean_score = np.mean(self.knn_scores)
        tree_mean_score = np.mean(self.tree_scores)
        softmax_mean_score = np.mean(self.softmax_scores)
        
        self.display_metrics(knn_metrics, tree_metrics, softmax_metrics, knn_mean_score, tree_mean_score, softmax_mean_score)
        
        self.knn_metrics_label.setVisible(True)
        self.tree_metrics_label.setVisible(True)
        self.softmax_metrics_label.setVisible(True)
        self.abort_button.setVisible(False)
        self.train_button.setEnabled(True)
    
    def display_metrics(self, knn_metrics, tree_metrics, softmax_metrics, knn_mean_score, tree_mean_score, softmax_mean_score):
        knn_metrics_text = 'KNN Metrics:\n' + \
                        f'accuracy: {knn_mean_score:.3f}\n' + \
                        '\n'.join([f'{metric}: {value:.3f}' for metric, value in knn_metrics.items()])
        
        tree_metrics_text = 'Classification Tree Metrics:\n' + \
                            f'accuracy: {tree_mean_score:.3f}\n' + \
                            '\n'.join([f'{metric}: {value:.3f}' for metric, value in tree_metrics.items()])
        
        softmax_metrics_text = 'Softmax Regression Metrics:\n' + \
                            f'accuracy: {softmax_mean_score:.3f}\n' + \
                            '\n'.join([f'{metric}: {value:.3f}' for metric, value in softmax_metrics.items()])
        
        self.knn_metrics_label.setText(knn_metrics_text)
        self.tree_metrics_label.setText(tree_metrics_text)
        self.softmax_metrics_label.setText(softmax_metrics_text)


