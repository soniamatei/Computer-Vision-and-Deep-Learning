import torch


def mean_pixel_accuracy(confusion_matrix: torch.Tensor) -> int:
    """
    Calculates the mean pixel accuracy based on a confusion matrix.
    @param confusion_matrix: confusion matrix for a prediction
    @return: the result
    """
    correct_pixels = torch.diag(confusion_matrix).sum()
    total_pixels = confusion_matrix.sum()

    pixel_accuracy = correct_pixels / total_pixels
    return pixel_accuracy.item()


def mean_intersection_over_union(confusion_matrix: torch.Tensor) -> int:
    """
    Calculates the mean intersection over union based on a confusion matrix.
    @param confusion_matrix: confusion matrix for a prediction
    @return: the result
    """
    intersection = torch.diag(confusion_matrix)
    ground_truth_per_class = confusion_matrix.sum(dim=1)
    predicted_per_class = confusion_matrix.sum(dim=0)

    union = ground_truth_per_class + predicted_per_class - intersection

    iou_per_class = intersection / union
    mean_iou = iou_per_class.mean()

    return mean_iou.item()
