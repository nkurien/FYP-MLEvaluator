from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton, QLineEdit
from PyQt5.QtGui import QIntValidator

class AdvancedSettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.setModal(True)

        layout = QVBoxLayout()

        # Number of folds setting
        folds_layout = QHBoxLayout()
        folds_label = QLabel("Number of Folds:")
        self.folds_spinbox = QSpinBox()
        self.folds_spinbox.setMinimum(2)
        self.folds_spinbox.setMaximum(20)
        self.folds_spinbox.setValue(settings["num_folds"])  # Set the current value
        folds_layout.addWidget(folds_label)
        folds_layout.addWidget(self.folds_spinbox)
        layout.addLayout(folds_layout)

        # Seed setting
        seed_layout = QHBoxLayout()
        seed_label = QLabel("Random Seed:")
        self.seed_input = QLineEdit()
        self.seed_input.setValidator(QIntValidator(0, 999999))  # Restrict input to integers up to 6 digits
        self.seed_input.setText(str(settings["seed"]))  # Set the current value
        seed_layout.addWidget(seed_label)
        seed_layout.addWidget(self.seed_input)
        layout.addLayout(seed_layout)

        # OK and Cancel buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def get_settings(self):
        settings = {
            "num_folds": self.folds_spinbox.value(),
            "seed": int(self.seed_input.text()),  # Convert the input to an integer
        }
        return settings