import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt
from preprocessing import load_dataset

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Dataset Loader')
        self.setGeometry(100, 100, 400, 300)  
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()

        file_layout = QHBoxLayout()
        self.file_label = QLabel('Dataset File:')
        self.file_text = QLineEdit()
        self.file_button = QPushButton('Browse')
        self.file_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_text)
        file_layout.addWidget(self.file_button)

        self.load_button = QPushButton('Load Dataset')
        self.load_button.clicked.connect(self.load_dataset)

        self.preview_label = QLabel('')
        self.preview_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.preview_label.setWordWrap(True)

        self.train_button = QPushButton('Train Models')
        self.train_button.setEnabled(False)
        self.train_button.clicked.connect(self.train_models)

        layout.addWidget(self.load_button)
        layout.addWidget(self.preview_label)  # Add the preview label to the layout
        layout.addLayout(file_layout)
        layout.addWidget(self.load_button)
        layout.addWidget(self.train_button)

        self.setLayout(layout)
    
    def browse_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Dataset File', '', 'All Files (*)', options=options)
        if file_path:
            self.file_text.setText(file_path)
    
    def load_dataset(self):
        file_path = self.file_text.text()
        if file_path:
            # Load the dataset using the load_dataset function
            X, y = load_dataset(file_path)
            QMessageBox.information(self, 'Dataset Loaded', 'Dataset has been loaded successfully.')
            self.train_button.setEnabled(True)

            # Display the first five lines of the loaded dataset
            preview_text = 'Dataset Preview:\n'
            for i in range(min(5, len(X))):
                preview_text += f'Line {i+1}: {X[i]}\n'
            self.preview_label.setText(preview_text)
        else:
            QMessageBox.warning(self, 'No File Selected', 'Please select a dataset file.')
    
    def train_models(self):
        # Train the models here using the loaded dataset
        QMessageBox.information(self, 'Training Complete', 'Models have been trained successfully.')
    

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()