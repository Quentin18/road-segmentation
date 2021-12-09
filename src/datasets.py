"""
Custom dataset classes for satellite images.
"""
import os
from typing import Callable, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split


class SatelliteImagesDataset(Dataset):
    """Dataset of satellite images."""
    def __init__(
        self,
        img_dir: str,
        gt_dir: str = None,
        image_transform: Callable = None,
        mask_transform: Callable = None,
    ) -> None:
        """Inits a satellite images dataset. It works with both training and
        testing datasets.

        Args:
            img_dir (str): path of images directory.
            gt_dir (str, optional): path of groundtruth directory.
            Defaults to None.
            image_transform (Callable, optional): optional transform to be
            applied on an image. Defaults to None.
            mask_transform (Callable, optional): optional transform to be
            applied on a mask. Defaults to None.
        """
        # Directories paths
        self.img_dir = img_dir
        self.gt_dir = gt_dir

        # Lists of images and masks names
        if self.gt_dir is not None:
            self.images_names = self._get_filenames(self.img_dir)
            self.masks_names = self._get_filenames(self.gt_dir)
        else:
            self.images_names = [f'{name}/{name}.png' for name in
                                 self._get_filenames(self.img_dir)]
            self.masks_names = list()

        # Transforms
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    @staticmethod
    def _get_filenames(dirname: str) -> List[str]:
        """Returns a sorted list of filenames from a directory.

        Args:
            dirname (str): path of the directory.

        Returns:
            List[str]: sorted list of filenames.
        """
        return sorted(os.listdir(dirname))

    @staticmethod
    def _read_img(img_path: str) -> Image:
        """Reads an image from its path.

        Args:
            img_path (str): image path.

        Returns:
            Image: PIL image.
        """
        return Image.open(img_path)

    def __len__(self) -> int:
        """Returns the number of images in the dataset.

        Returns:
            int: number of images.
        """
        return len(self.images_names)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Gets an image and its corresponding mask.
        If no masks in the dataset, it returns `(image, 0)`.

        Args:
            index (int): index of the image in the list.

        Returns:
            Tuple[np.ndarray, np.ndarray]: image, mask.
        """
        img_path = os.path.join(self.img_dir, self.images_names[index])
        if self.gt_dir:
            mask_path = os.path.join(self.gt_dir, self.masks_names[index])

        image = self._read_img(img_path)
        mask = self._read_img(mask_path) if self.gt_dir else 0

        # Apply image transformations
        if self.image_transform is not None:
            image = self.image_transform(image)

        # Apply mask transformations
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask

    def get_img_filename(self, index: int) -> str:
        """Returns the filename of an image from the index.

        Args:
            index (int): index of the image in the list.

        Returns:
            str: filename corresponding to the index.
        """
        return self.images_names[index]


def train_test_split(
    dataset: Dataset,
    test_ratio: float,
    seed: int = None,
) -> Tuple[Dataset, Dataset]:
    """Splits a dataset into random train and test subsets.

    Args:
        dataset (Dataset): dataset.
        test_ratio (float): test proportion (between 0 and 1).
        seed (int, optional): seed. Defaults to None.

    Returns:
        Tuple[Dataset, Dataset]: train and test datasets.
    """
    # Define generator
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    # Define lengths of subsets
    train_ratio = 1 - test_ratio
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    lengths = [train_size, test_size]

    # Split
    train_dataset, test_dataset = random_split(dataset, lengths, generator)

    return train_dataset, test_dataset
