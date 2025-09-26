from sklearn.metrics import precision_score, recall_score, accuracy_score


def compute_evaluation_metrics(y_true, y_pred):
    """
    Args:
        y_true: true labels
        y_pred: predicted labels
    Returns:
        precision: precision score
        recall: recall score
        accuracy: accuracy score
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return p, r, acc