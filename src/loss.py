"""
Custom loss functions.

Source:
https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
"""
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Criterion that computes Sørensen-Dice Coefficient loss.

    https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    """
    def __init__(self):
        super().__init__()
        self.smooth = 1.0

    def forward(self, input, target):
        # Flatten label and prediction tensors
        input = input.view(-1)
        target = target.view(-1)

        intersection = (input * target).sum()
        dice = (2. * intersection + self.smooth) / (
            input.sum() + target.sum() + self.smooth)

        return 1 - dice
