import numpy as np
from sklearn import metrics


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes = None) -> np.ndarray:
    """
    Computes the confusion matrix from labels (y_true) and predictions (y_pred).
    The matrix columns represent the prediction labels and the rows represent the ground truth labels.
    The confusion matrix is always a 2-D array of shape `[num_classes, num_classes]`,
    where `num_classes` is the number of valid labels for a given classification task.
    The arguments y_true and y_pred must have the same shapes in order for this function to work

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    conf_mat = None
    # TODO your code here - compute the confusion matrix
    # even here try to use vectorization, so NO for loops

    # 0. if the number of classes is not provided, compute it based on the y_true and y_pred arrays
    if num_classes is None:
        num_classes = np.unique(np.concatenate([y_true, y_pred])).size
    # 1. create a confusion matrix of shape (num_classes, num_classes) and initialize it to 0
    conf_mat = np.zeros((num_classes, num_classes))
    # 2. use argmax to get the maximal prediction for each sample
    # hint: you might find np.add.at useful: https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
    np.add.at(conf_mat, (y_true, y_pred), 1)
    # end TODO your code here
    return conf_mat


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes=None) -> float:
    """
    Computes the precision score.
    For binary classification, the precision score is defined as the ratio tp / (tp + fp)
    where tp is the number of true positives and fp the number of false positives.

    For multiclass classification, the precision and recall scores are obtained by summing over the rows / columns
    of the confusion matrix.

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    precision = 0
    # TODO your code here
    if num_classes is None:
        num_classes = np.unique(np.concatenate([y_true, y_pred])).size

    conf_mat = confusion_matrix(y_true, y_pred, num_classes)

    for i in range(num_classes):
        tp = conf_mat[i, i]
        fp = np.sum(conf_mat[:, i]) - tp
        precision += tp / (tp + fp) if (tp + fp) != 0 else 0.0

    precision /= num_classes
    # end TODO your code here
    return precision


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes=None)  -> float:
    """
    Computes the recall score.
    For binary classification, the recall score is defined as the ratio tp / (tp + fn)
    where tp is the number of true positives and fn the number of false negatives

    For multiclass classification, the precision and recall scores are obtained by summing over the rows / columns
    of the confusion matrix.

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    recall = 0
    # TODO your code here
    if num_classes is None:
        num_classes = np.unique(np.concatenate([y_true, y_pred])).size

    conf_mat = confusion_matrix(y_true, y_pred, num_classes)

    for i in range(num_classes):
        tp = conf_mat[i, i]
        fn = np.sum(conf_mat[i, :]) - tp
        recall += tp / (tp + fn) if (tp + fn) != 0 else 0.0

    recall /= num_classes
    # end TODO your code here
    return recall


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # TODO your code here
    # remember, use vectorization, so no for loops
    # hint: you might find np.trace useful here https://numpy.org/doc/stable/reference/generated/numpy.trace.html
    # Sum of diagonal elements represents the total number of correct predictions
    num_classes = np.unique(np.concatenate([y_true, y_pred])).size
    conf_mat = confusion_matrix(y_true, y_pred, num_classes)

    correct_predictions = np.trace(conf_mat)
    total_samples = np.sum(conf_mat)
    acc_score = correct_predictions / total_samples if total_samples != 0 else 0.0

    # end TODO your code here
    return acc_score


if __name__ == '__main__':
    # TODO your tests here
    # add some test for your code.
    # you could use the sklearn.metrics module (with macro averaging to check your results)

    # Generate some example data for testing
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 2, 2])
    y_pred = np.array([0, 2, 1, 0, 2, 1, 0, 0, 2])

    # Calculate your metrics
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Precision Score:", precision_score(y_true, y_pred))
    print("Recall Score:", recall_score(y_true, y_pred))
    print("Accuracy Score:", accuracy_score(y_true, y_pred))

    # Compare with scikit-learn metrics
    print("\nScikit-learn Metrics:")
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_true, y_pred))
    print("Precision Score:", metrics.precision_score(y_true, y_pred, average='macro'))
    print("Recall Score:", metrics.recall_score(y_true, y_pred, average='macro'))
    print("Accuracy Score:", metrics.accuracy_score(y_true, y_pred))
