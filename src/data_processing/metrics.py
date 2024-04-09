import numpy as np
from data_processing.preprocessing import LabelEncoder

def calculate_confusion_matrix(y_true, y_pred, classes):
    """
    Manually calculate the confusion matrix for given true and predicted labels.
    
    Parameters:
    - y_true: array-like, true labels
    - y_pred: array-like, predicted labels
    - classes: list of unique classes
    
    Returns:
    - Confusion matrix as a 2D numpy array
    """
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for i, true_label in enumerate(classes):
        for j, pred_label in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    return cm

def calculate_metrics_from_cm(cm):
    """
    Calculate precision, recall, f1 score, and specificity from confusion matrix.
    
    Parameters:
    - cm: Confusion matrix
    
    Returns:
    - Dictionary of calculated metrics
    """
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (FP + FN + TP)
    
    # Avoid division by zero using np.divide with the where parameter
    precision = np.divide(TP, TP + FP, where=((TP + FP) != 0))
    recall = np.divide(TP, TP + FN, where=((TP + FN) != 0))
    
    # Handle F1 score calculation with potential zero denominator
    f1_score = 2 * (precision * recall) / (precision + recall + (precision + recall == 0))
    
    specificity = np.divide(TN, TN + FP, where=((TN + FP) != 0))
    
    # Use nanmean to calculate the mean while ignoring NaN values
    return {
        'precision': np.nanmean(precision),
        'recall': np.nanmean(recall),
        'f1_score': np.nanmean(f1_score),
        'specificity': np.nanmean(specificity)
    }

def calculate_metrics(y_true, y_pred):
    """
    Calculate precision, recall, F1 score, and specificity.
    Handles both binary and multi-class classification.
    
    Parameters:
    - y_true: array-like of shape (n_samples,) True labels.
    - y_pred: array-like of shape (n_samples,) Predicted labels.
    
    Returns:
    - metrics: dict containing precision, recall, F1 score, and specificity.
    """
    # Convert string inputs to numeric using NumericConverter
    converter = LabelEncoder()
    y_true = converter.fit_transform(y_true)
    y_pred = converter.transform(y_pred)
    
    classes = np.unique(np.concatenate((y_true, y_pred)))
    cm = calculate_confusion_matrix(y_true, y_pred, classes)
    metrics = calculate_metrics_from_cm(cm)
    return metrics

# Example usage
if __name__ == "__main__":
    y_true = np.array([1, 2, 3, 4, 2, 3])
    y_pred = np.array([1, 2, 3, 3, 2, 3])
    metrics = calculate_metrics(y_true, y_pred)
    print(metrics)
