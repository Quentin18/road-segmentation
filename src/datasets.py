"""
Custom dataset classes for satellite images.

https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
"""
import os
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, random_split


class SatelliteImagesTrainDataset(Dataset):
    """Training dataset of satellite images."""
    def __init__(
        self,
        root_dir: str,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None
    ) -> None:
        """Inits the satellite images training dataset.

        Args:
            root_dir (str): path of the directory with the `images` and
            `groundtruth` directories.
            image_transform (Callable, optional): optional transform to be
            applied on an image. Defaults to None.
            mask_transform (Callable, optional): optional transform to be
            applied on a mask. Defaults to None.
        """
        # Directories paths
        self.img_dir = os.path.join(root_dir, 'images')
        self.gt_dir = os.path.join(root_dir, 'groundtruth')

        # Lists of images and masks names
        self.images_names = os.listdir(self.img_dir)
        self.masks_names = os.listdir(self.gt_dir)

        # Transforms
        self.image_transform = image_transform
        self.mask_transform = mask_transform

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

        Args:
            index (int): index of the image in the list.

        Returns:
            Tuple[np.ndarray, np.ndarray]: image, mask.
        """
        img_path = os.path.join(self.img_dir, self.images_names[index])
        mask_path = os.path.join(self.gt_dir, self.masks_names[index])

        image = self._read_img(img_path)
        mask = self._read_img(mask_path)

        # Apply image transformations
        if self.image_transform is not None:
            image = self.image_transform(image)

        # Apply mask transformations
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask


class SatelliteImagesTestDataset(Dataset):
    """Testing dataset of satellite images."""
    def __init__(
        self,
        root_dir: str,
        image_transform: Optional[Callable] = None,
    ) -> None:
        """Inits the satellite images training dataset.

        Args:
            root_dir (str): path of the directory with the `images` and
            `groundtruth` directories.
            image_transform (Callable, optional): optional transform to be
            applied on an image. Defaults to None.
        """
        # Directories paths
        self.img_dir = os.path.join(root_dir, 'images')

        # Lists of images and masks names
        self.images_names = os.listdir(self.img_dir)

        # Transforms
        self.image_transform = image_transform

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
        """Gets an image

        Args:
            index (int): index of the image in the list.

        Returns:
            Tuple[np.ndarray]: image
        """
        img_path = os.path.join(self.img_dir, self.images_names[index])

        image = self._read_img(img_path)

        # Apply image transformations
        if self.image_transform is not None:
            image = self.image_transform(image)

        return image


def train_test_split(dataset: Dataset,
                     test_ratio: float) -> Tuple[Dataset, Dataset]:
    """Splits a dataset into random train and test subsets.

    Args:
        dataset (Dataset): dataset.
        test_ratio (float): test proportion (between 0 and 1).

    Returns:
        Tuple[Dataset, Dataset]: train and test datasets.
    """
    train_ratio = 1 - test_ratio
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    lengths = [train_size, test_size]
    train_dataset, test_dataset = random_split(dataset, lengths)
    return train_dataset, test_dataset
