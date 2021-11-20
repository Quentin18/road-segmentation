"""
Custom dataset classes for satellite images.

https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
"""
import os
from typing import Optional, Callable, Tuple

import numpy as np

from torch.utils.data import Dataset
from skimage import io


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
            image_transform (Optional[Callable], optional): optional transform
            to be applied on an image. Defaults to None.
            mask_transform (Optional[Callable], optional): optional transform
            to be applied on a mask. Defaults to None.
        """
        self.img_dir = os.path.join(root_dir, 'images')
        self.gt_dir = os.path.join(root_dir, 'groundtruth')
        self.images_names = os.listdir(self.img_dir)    # list of images names
        self.masks_names = os.listdir(self.gt_dir)      # list of masks names
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    @staticmethod
    def _read_img(img_name: str) -> np.ndarray:
        return io.imread(img_name)

    def __len__(self) -> int:
        return len(self.images_names)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = os.path.join(self.img_dir, self.images_names[index])
        mask_path = os.path.join(self.gt_dir, self.masks_names[index])

        image = self._read_img(img_path)
        mask = self._read_img(mask_path)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask


class SatelliteImagesTestDataset(Dataset):
    """Testing dataset of satellite images."""
    pass
