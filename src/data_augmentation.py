"""
Data augmentation functions.
"""
import os
import shutil

import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import FiveCrop, Resize
from tqdm import trange

from src.path import (DATA_TRAIN_AUG_GT_PATH, DATA_TRAIN_AUG_IMG_PATH,
                      DATA_TRAIN_AUG_PATH, DATA_TRAIN_GT_PATH,
                      DATA_TRAIN_IMG_PATH)

# Size of the new images
AUG_IMG_SIZE = 256

# Angles for rotations
ANGLES = [-90, -45, 45, 90]


class MultipleRotationCrop:
    """Rotate the image by the given angles and center crop."""
    def __init__(self, angles: list, size: int):
        self.angles = angles
        self.size = size

    def __call__(self, img):
        return [
            TF.center_crop(TF.rotate(img, angle), self.size)
            for angle in self.angles
        ]


def create_augmented_dataset(replace: bool = False) -> None:
    """Creates the augmented dataset.

    It contains 1000 images of size 256x256:
    - Resized images from train data (100)
    - Cropped images into four corners and the central crop (5x100)
    - Rotated and cropped images (4x100)

    Args:
        replace (bool, optional): True to replace images if already exist.
        Defaults to False.
    """
    # Ignore if dataset already created
    if os.path.exists(DATA_TRAIN_AUG_PATH) and not replace:
        return

    # Reset directory
    shutil.rmtree(DATA_TRAIN_AUG_PATH, ignore_errors=True)

    # Creates directories
    for dirname in (DATA_TRAIN_AUG_GT_PATH, DATA_TRAIN_AUG_IMG_PATH):
        os.makedirs(dirname, exist_ok=True)

    # Get images paths
    images_names = sorted(os.listdir(DATA_TRAIN_IMG_PATH))
    masks_names = sorted(os.listdir(DATA_TRAIN_GT_PATH))

    # Define transforms
    size = (AUG_IMG_SIZE, AUG_IMG_SIZE)
    resize = Resize(size)
    five_crop = FiveCrop(size)
    multiple_rotate_crop = MultipleRotationCrop(ANGLES, size)

    # Number of crops by image
    nb_crop = 5

    # Number of rotations by image
    nb_rot = len(ANGLES)

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

            # 3. Rotate and crop images
            images_rot = multiple_rotate_crop(image)
            masks_rot = multiple_rotate_crop(mask)

            # Save rotated images
            for k in range(nb_rot):
                image_rot = images_rot[k]
                mask_rot = masks_rot[k]

                filename = f'satImage_rot{i * nb_rot + k}.png'
                image_rot_path = os.path.join(
                    DATA_TRAIN_AUG_IMG_PATH, filename
                )
                mask_rot_path = os.path.join(
                    DATA_TRAIN_AUG_GT_PATH, filename
                )
                image_rot.save(image_rot_path)
                mask_rot.save(mask_rot_path)
