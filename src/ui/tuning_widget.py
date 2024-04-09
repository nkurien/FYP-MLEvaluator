import seaborn as sns
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout

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
        ax2.set_xlabel('Min Leaf Size')
        ax2.set_ylabel('Max Depth')

        # Plot Softmax Regression heatmap
        softmax_scores_table = self.reshape_scores(softmax_grid_search.scores_, softmax_grid_search.param_grid)
        sns.heatmap(softmax_scores_table, annot=True, cmap='coolwarm', fmt='.3f',
                    xticklabels=softmax_grid_search.param_grid['learning_rate'],
                    yticklabels=softmax_grid_search.param_grid['n_iterations'],
                    ax=ax3, cbar_kws={'label': 'Score'})
        ax3.set_title('Softmax Regression Tuning Results')
        ax3.set_xlabel('Learning Rate')
        ax3.set_ylabel('N Iterations')

        self.figure.tight_layout()
        self.canvas.draw()

    def reshape_scores(self, scores, param_grid):
        # Reshape scores into a 2D array based on parameter grid
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        scores_table = np.reshape(scores, (len(param_values[0]), len(param_values[1])))
        return scores_table