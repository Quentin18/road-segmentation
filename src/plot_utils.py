"""
Plots utils using matplotlib.
"""
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image
from torch import Tensor
from torchvision import transforms


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def concatenate_images(img, gt_img):
    """Concatenate an image and its groundtruth."""
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def plot_loss(train_loss: list, test_loss: list, path: str = None) -> None:
    """Plots train and test loss.

    Args:
        train_loss (list): train loss list.
        test_loss (list): test loss list.
        path(str, optional): path to save the figure.
    """
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(train_loss, label='Train loss')
    plt.plot(test_loss, label='Test loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    if path is not None:
        plt.savefig(path)


def plot_validation_F1(F1_score: list, threshold: list, optimum: int,
                       path: str = None) -> None:
    """Plots train and test loss.

    Args:
        F1_score (list): F1_score list.
        threshold (list): threshold of validation list.
        optimum (int): index of the optimum threshold
        path (str, optional): path to save the figure.
    """
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(threshold, F1_score, label='F1 score')
    plt.plot(threshold[optimum], F1_score[optimum], marker="*", color="red",
             label='optimum')
    plt.title('Validation on threshold parameter with F1 score')
    plt.xlabel('Threshold')
    plt.ylabel('F1 score')
    plt.legend(loc='lower left')
    if path is not None:
        plt.savefig(path)


def plot_images(
    image: Union[Image, Tensor],
    mask: Union[Image, Tensor],
    pred: Union[Image, Tensor] = None,
    show: bool = True,
) -> None:
    """Plots an image, its mask and optionaly a predicted mask.

    Args:
        image (Union[Image, Tensor]): image as PIL image of tensor.
        mask (Union[Image, Tensor]): mask as PIL image or tensor.
        pred (Union[Image, Tensor], optional): predicted mask as PIL image or
        tensor. Defaults to None.
        show (bool): call show.
    """
    # Convert image, mask and pred to PIL images
    transform = transforms.ToPILImage()
    if isinstance(image, Tensor):
        image = transform(image)
    if isinstance(mask, Tensor):
        mask = transform(mask)
    if pred is not None and isinstance(pred, Tensor):
        pred = transform(pred)

    # Create figure
    ncols = 2 if pred is None else 3
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(10, 5))

    # Plot image and mask
    images, titles = [image, mask], ['Image', 'Mask']
    if pred is not None:
        images.append(pred)
        titles[1] = 'Original mask'
        titles.append('Predicted mask')

    for i, (img, title) in enumerate(zip(images, titles)):
        ax[i].imshow(img)
        ax[i].set_title(title)
        ax[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    fig.tight_layout()

    if show:
        plt.show()
