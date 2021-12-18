"""
Data augmentation functions.
"""
import os
import shutil

import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import Resize, TenCrop
from tqdm import trange

from src.path import (DATA_TRAIN_AUG_GT_PATH, DATA_TRAIN_AUG_IMG_PATH,
                      DATA_TRAIN_AUG_PATH, DATA_TRAIN_GT_PATH,
                      DATA_TRAIN_IMG_PATH)

# Size of the base images
BASE_IMG_SIZE = (400, 400)

# Size of the new images
AUG_IMG_SIZE = (256, 256)

# Angles for rotations
ANGLES = [45 * (i + 1) for i in range(7)]


class MultipleRotationCrop:
    """Rotate the image by the given angles and center crop."""
    def __init__(self, angles: list, size: tuple):
        self.angles = angles
        self.size = size

    def __call__(self, img):
        return [
            TF.center_crop(TF.rotate(img, angle), self.size)
            for angle in self.angles
        ]


def save_aug_img_mask(image: Image, mask: Image, num: int) -> None:
    """Saves an image and a mask in the augmented directories.

    Args:
        image (Image): image.
        mask (Image): mask.
        num (int): number of the image/mask.
    """
    filename = f'satImage_{num:04d}.png'
    image_path = os.path.join(DATA_TRAIN_AUG_IMG_PATH, filename)
    mask_path = os.path.join(DATA_TRAIN_AUG_GT_PATH, filename)
    image.save(image_path)
    mask.save(mask_path)


def create_augmented_dataset(replace: bool = False) -> None:
    """Creates the augmented dataset.

    It contains 1800 images of size 256x256:
    - Resize images from train data (100)
    - Crop images into four corners and the central crop plus the flipped
    version of these (100x10)
    - Rotate images and center crop (100x7)

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
    resize = Resize(size=AUG_IMG_SIZE)
    ten_crop = TenCrop(size=AUG_IMG_SIZE)
    multiple_rotation_crop = MultipleRotationCrop(
        angles=ANGLES,
        size=AUG_IMG_SIZE,
    )

    # Counter for the number of the image/mask
    num = 1

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
            save_aug_img_mask(image_resized, mask_resized, num)
            num += 1

            # 2. Crop images into four corners and the central crop plus the
            # flipped version of these
            images_cropped = ten_crop(image)
            mask_cropped = ten_crop(mask)

            # Save cropped images
            for i, m in zip(images_cropped, mask_cropped):
                save_aug_img_mask(i, m, num)
                num += 1

            # 3. Rotate images and center crop
            images_rotated = multiple_rotation_crop(image)
            mask_rotated = multiple_rotation_crop(mask)

            # Save rotated images
            for i, m in zip(images_rotated, mask_rotated):
                save_aug_img_mask(i, m, num)
                num += 1
