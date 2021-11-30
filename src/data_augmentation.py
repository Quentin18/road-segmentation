"""
Data augmentation functions.
"""
import os

import torch
from PIL import Image
from torchvision import transforms
from tqdm import trange

from src.path import (DATA_TRAIN_AUG_GT_PATH, DATA_TRAIN_AUG_IMG_PATH,
                      DATA_TRAIN_AUG_PATH, DATA_TRAIN_GT_PATH,
                      DATA_TRAIN_IMG_PATH)


def create_augmented_dataset(img_size: int = 280,
                             replace: bool = False) -> None:
    """Creates the augmented dataset.

    Args:
        img_size (int, optional): size of the output images. Defaults to 280.
        replace (bool, optional): True to replace images if already exist.
        Defaults to False.
    """
    # Ignore if dataset already created
    if os.path.exists(DATA_TRAIN_AUG_PATH) and not replace:
        return

    # Creates directories
    for dirname in (DATA_TRAIN_AUG_GT_PATH, DATA_TRAIN_AUG_IMG_PATH):
        os.makedirs(dirname, exist_ok=True)

    # Get images paths
    images_names = sorted(os.listdir(DATA_TRAIN_IMG_PATH))
    masks_names = sorted(os.listdir(DATA_TRAIN_GT_PATH))

    # Define transforms
    size = (img_size, img_size)
    resize = transforms.Resize(size)
    five_crop = transforms.FiveCrop(size)
    rotate_crop = transforms.Compose([
        transforms.RandomRotation((-180, 180)),
        transforms.CenterCrop(size),
    ])

    # Number of crops by image
    nb_crop = 5

    # Number of rotations by image
    nb_rot = 5

    # Create augmented dataset
    with trange(len(images_names), unit='image') as t:
        for i in t:
            # Retrieve images names
            image_name = images_names[i]
            mask_name = masks_names[i]

            t.set_description(desc=image_name)

            # Get images paths
            image_path = os.path.join(DATA_TRAIN_IMG_PATH, image_name)
            mask_path = os.path.join(DATA_TRAIN_GT_PATH, mask_name)

            # Open images
            image = Image.open(image_path)
            mask = Image.open(mask_path)

            # 1. Resize images
            image_resized = resize(image)
            mask_resized = resize(mask)

            # Save resized images
            image_resized_path = os.path.join(
                DATA_TRAIN_AUG_IMG_PATH, image_name
            )
            mask_resized_path = os.path.join(
                DATA_TRAIN_AUG_GT_PATH, mask_name
            )
            image_resized.save(image_resized_path)
            mask_resized.save(mask_resized_path)

            # 2. Crop the images into four corners and the central crop
            images_crop = five_crop(image)
            masks_crop = five_crop(mask)

            # Save cropped images
            for k in range(nb_crop):
                image_crop = images_crop[k]
                mask_crop = masks_crop[k]

                filename = f'satImage_crop{i * nb_crop + k}.png'
                image_crop_path = os.path.join(
                    DATA_TRAIN_AUG_IMG_PATH, filename
                )
                mask_crop_path = os.path.join(
                    DATA_TRAIN_AUG_GT_PATH, filename
                )
                image_crop.save(image_crop_path)
                mask_crop.save(mask_crop_path)

            # 3. Rotate and crop images (set seed to have same transform)
            for k in range(nb_rot):
                seed = i * nb_rot + k
                torch.manual_seed(seed)
                image_rot = rotate_crop(image)
                torch.manual_seed(seed)
                mask_rot = rotate_crop(mask)

                # Save rotated images
                filename = f'satImage_rot{seed}.png'
                image_rot_path = os.path.join(
                    DATA_TRAIN_AUG_IMG_PATH, filename
                )
                mask_rot_path = os.path.join(
                    DATA_TRAIN_AUG_GT_PATH, filename
                )
                image_rot.save(image_rot_path)
                mask_rot.save(mask_rot_path)
