import numpy as np

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
    
    precision = np.mean(TP / (TP + FP))
    recall = np.mean(TP / (TP + FN))
    f1_score = 2 * (precision * recall) / (precision + recall)
    specificity = np.mean(TN / (TN + FP))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity
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
