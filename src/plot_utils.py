"""
Plots utils using matplotlib.
"""
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm


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
        path(str, optional): path to save the figure. Defaults to None.
    """
    plt.style.use('ggplot')
    plt.plot(train_loss, label='Train loss')
    plt.plot(test_loss, label='Test loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    if path is not None:
        plt.savefig(path)


def plot_history(epoch_metrics: dict, path: str = None) -> None:
    """Plots metrics from an history.

    Args:
        epoch_metrics (dict): dictionary of metrics.
        path(str, optional): path to save the figure. Defaults to None.
    """
    plt.style.use('ggplot')

    # Create figure
    fig, (ax_loss, ax_accuracy_f1) = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 5)
    )

    # Plot metrics
    for key, values in epoch_metrics.items():
        ax = ax_loss if 'loss' in key else ax_accuracy_f1
        ax.plot(values, label=key.replace('_', ' ').capitalize())

    # Set labels
    ax_loss.set(title='Loss', xlabel='Epoch', ylabel='Loss')
    ax_loss.legend()
    ax_accuracy_f1.set(title='Metrics', xlabel='Epoch', ylabel='Accuracy / F1')
    ax_accuracy_f1.legend()

    fig.tight_layout()

    # Save figure
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
    image: Union[Image.Image, Tensor],
    mask: Union[Image.Image, Tensor],
    pred: Union[Image.Image, Tensor] = None,
    path: str = None,
) -> None:
    """Plots an image, its mask and optionaly a predicted mask.

    Args:
        image (Union[Image, Tensor]): image as PIL image of tensor.
        mask (Union[Image, Tensor]): mask as PIL image or tensor.
        pred (Union[Image, Tensor], optional): predicted mask as PIL image or
        tensor. Defaults to None.
        path(str, optional): path to save the figure. Defaults to None.
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

    # Save figure
    if path is not None:
        plt.savefig(path)
        plt.close(fig)


def make_img_overlay(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Convert an image an its mask as numpy arrays to an overlay.

    Args:
        image (np.ndarray): image as numpy array.
        mask (np.ndarray): mask as numpy array.

    Returns:
        Image.Image: overlay.
    """
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[:, :, 0] = mask

    img8 = img_float_to_uint8(image)
    background = Image.fromarray(img8, 'RGB').convert('RGBA')
    overlay = Image.fromarray(color_mask, 'RGB').convert('RGBA')
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def make_img_overlays(image_dir: str, mask_dir: str, output_dir: str) -> None:
    """Makes the overlays from predictions.

    Args:
        image_dir (str): images directory.
        mask_dir (str): masks directory.
        output_dir (str): output directory.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get image names
    image_names = [f'test_{i + 1}/test_{i + 1}.png' for i in range(50)]
    mask_names = sorted(os.listdir(mask_dir))

    for image_name, mask_name in tqdm(
        zip(image_names, mask_names), total=len(image_names)
    ):
        # Get paths
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, mask_name)

        # Read images as numpy arrays
        image = np.asarray(Image.open(image_path))
        mask = np.asarray(Image.open(mask_path))

        # Make overlay
        overlay = make_img_overlay(image, mask)

        # Save overlay
        overlay.save(os.path.join(output_dir, mask_name))
