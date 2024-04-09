from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
