"""
Metrics functions.
"""
import torch
from sklearn.metrics import accuracy_score, f1_score


def accuracy_score_tensors(
    target: torch.Tensor,
    output: torch.Tensor,
) -> float:
    """Accuracy classification score from tensors.

    Args:
        target (torch.Tensor): Ground truth (correct) labels.
        output (torch.Tensor): Predicted labels, as returned by a classifier.

    Returns:
        float: accuracy score between 0 and 1.
    """
    target_flatten = torch.flatten(target)
    output_flatten = torch.flatten(output)
    return accuracy_score(target_flatten, output_flatten, normalize=True)


def f1_score_tensors(
    target: torch.Tensor,
    output: torch.Tensor,
) -> float:
    """F1 score from tensors.

    Args:
        target (torch.Tensor): Ground truth (correct) labels.
        output (torch.Tensor): Predicted labels, as returned by a classifier.

    Returns:
        float: f1 score between 0 and 1.
    """
    target_flatten = torch.flatten(target)
    output_flatten = torch.flatten(output)
    return f1_score(target_flatten, output_flatten)
