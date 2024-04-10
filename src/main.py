import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from ui.ui import MainWindow  # Assuming 'ui' is a package and 'main_window' is a module within it


def main():

    # Create an instance of QApplication
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('/Users/nathan/Desktop/PROJECT/src/ui/resources/icon.png'))
    app.setApplicationName("your title")

    # Create an instance of your application's main window
    main_window = MainWindow()

    # Show the main window
    main_window.show()

    # Start the application's event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()