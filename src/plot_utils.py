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


def plot_loss(train_loss: list, test_loss: list) -> None:
    """Plots train and test loss.

    Args:
        train_loss (list): train loss list.
        test_loss (list): test loss list.
    """
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(train_loss, label='Train loss')
    plt.plot(test_loss, label='Test loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')


def plot_images(
    image: Union[Image, Tensor],
    mask: Union[Image, Tensor],
    pred: Union[Image, Tensor] = None
) -> None:
    """Plots an image, its mask and optionaly a predicted mask.

    Args:
        image (Union[Image, Tensor]): image as PIL image of tensor.
        mask (Union[Image, Tensor]): mask as PIL image or tensor.
        pred (Union[Image, Tensor]): predicted mask as PIL image or tensor.
        Defaults to None.
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
    plt.figure(figsize=(10, 10))
    ncols = 2 if pred is None else 3
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(10, 10))

    # Plot image and mask
    images, titles = [image, mask], ['Image', 'Mask']
    if pred is not None:
        images.append(pred)
        titles[1] = 'Original mask'
        titles.append('Predicted mask')

    for i, (img, title) in enumerate(zip(images, titles)):
        ax[i].imshow(img)
        ax[i].set_title(title)

    fig.tight_layout()
