"""
Functions to manage submissions for AICrowd.
"""
import os
import re
from typing import List

import numpy as np
from PIL import Image
from skimage import io
from tqdm import tqdm, trange

from src.path import OUT_DIR


def binary_to_uint8(array: np.ndarray) -> np.ndarray:
    """Converts an array of binary labels to a uint8.

    Args:
        array (np.ndarray): array of binary labels.

    Returns:
        np.ndarray: uint8 array.
    """
    return (array * 255).round().astype(np.uint8)


def get_submission_lines(submission_filename: str) -> List[str]:
    """Returns the lines of a submission file.

    Args:
        submission_filename (str): path of the csv submission file.

    Returns:
        List[str]: lines of the submission file.
    """
    with open(submission_filename, 'r') as f:
        lines = f.readlines()
    return lines


def submission_to_mask(submission_filename: str, image_id: int,
                       mask_filename: str = None,
                       w: int = 16, h: int = 16) -> np.ndarray:
    """Returns a mask from a submission file and its id.

    Args:
        submission_filename (str): submission csv file path.
        image_id (int): image id.
        mask_filename (str, optional): mask file path. Defaults to None.
        w (int, optional): width. Defaults to 16.
        h (int, optional): height. Defaults to 16.

    Returns:
        np.ndarray: mask.
    """
    # Get submission lines
    lines = get_submission_lines(submission_filename)

    # Init image
    img_width = int(np.ceil(600 / w) * w)
    img_height = int(np.ceil(600 / h) * h)
    im = np.zeros((img_width, img_height), dtype=np.uint8)
    image_id_str = f'{image_id:03d}_'

    # Fill image
    for line in lines[1:]:
        if image_id_str not in line:
            continue

        tokens = line.split(',')
        id_, prediction = tokens[0], int(tokens[1])
        tokens = id_.split('_')
        i, j = int(tokens[1]), int(tokens[2])

        je = min(j + w, img_width)
        ie = min(i + h, img_height)
        adata = np.zeros((w, h)) if prediction == 0 else np.ones((w, h))
        im[j:je, i:ie] = binary_to_uint8(adata)

    # Save mask
    if mask_filename is not None:
        Image.fromarray(im).save(mask_filename)

    return im


def submission_to_masks(submission_filename: str, nb_masks: int = 50,
                        masks_dirname: str = None,
                        w: int = 16, h: int = 16) -> List[np.ndarray]:
    """Returns the list of masks corresponding to a submission.

    Args:
        submission_filename (str): submission csv file path.
        nb_masks (int, optional): number of masks to create. Defaults to 50.
        masks_dirname (str, optional): directory of masks saved as images.
        Defaults to None.
        w (int, optional): width. Defaults to 16.
        h (int, optional): height. Defaults to 16.

    Returns:
        List[np.ndarray]: list of masks.
    """
    masks = list()

    # Create masks directory
    if masks_dirname is not None and not os.path.exists(masks_dirname):
        os.mkdir(masks_dirname)

    # Create masks
    for i in trange(nb_masks):
        image_id = i + 1
        if masks_dirname is not None:
            mask_name = f'prediction_{image_id:03d}.png'
            mask_filename = os.path.join(masks_dirname, mask_name)
        else:
            mask_filename = None
        mask = submission_to_mask(submission_filename, image_id, mask_filename,
                                  w, h)
        masks.append(mask)

    return masks


def patch_to_label(patch: np.ndarray,
                   foreground_threshold: float = 0.25) -> int:
    """Assigns a label to a patch.

    Args:
        patch (np.ndarray): patch.
        foreground_threshold (float, optional): foreground_threshold.
        Defaults to 0.25.

    Returns:
        int: 0 or 1.
    """
    return int(np.mean(patch) > (foreground_threshold * 255))


def mask_to_submission_strings(mask_filename: str, patch_size: int = 16,
                               foreground_threshold: float = 0.25,
                               clean: bool = False):
    """Reads a single mask image and outputs the strings that should go into
    the submission file.

    Args:
        mask_filename (str): mask file path.
        patch_size (int, optional): patch size. Defaults to 16.
        foreground_threshold (float, optional): foreground_threshold.
        Defaults to 0.25.
        clean (bool, optional): clean the patches by a neighbor method.
        Defaults to False.
    """
    mask_name = os.path.basename(mask_filename)
    img_number = int(re.search(r"\d+", mask_name).group(0))
    im = io.imread(os.path.join(OUT_DIR, 'submission', mask_filename))

    # Create mask of patch 38x38
    mask_patch = np.zeros(
        shape=(im.shape[0] // patch_size, im.shape[1] // patch_size),
        dtype=np.uint8,
    )
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(
                patch=patch,
                foreground_threshold=foreground_threshold,
            )
            mask_patch[j // patch_size, i // patch_size] = label

    # Improve patches
    if clean:
        mask_patch_clean = np.copy(mask_patch)
        for j in range(2, mask_patch.shape[1] - 2):
            for i in range(2, mask_patch.shape[0] - 2):
                label = mask_patch[j, i]

                # If not road: ignore
                if label == 0:
                    continue

                if mask_patch[j - 2, i]:
                    mask_patch_clean[j - 1, i] = 1

                if mask_patch[j, i - 2]:
                    mask_patch_clean[j, i - 1] = 1

                if mask_patch[j + 2, i]:
                    mask_patch_clean[j + 1, i] = 1

                if mask_patch[j, i + 2]:
                    mask_patch_clean[j, i + 1] = 1

        mask_patch = mask_patch_clean

    # Yield patches
    for j in range(mask_patch.shape[1]):
        for i in range(mask_patch.shape[0]):
            label = mask_patch[j, i]
            yield f'{img_number:03d}_{j * patch_size}_{i * patch_size},{label}'


def masks_to_submission(submission_filename: str,
                        masks_filenames: list, patch_size: int = 16,
                        foreground_threshold: float = 0.25,
                        clean: bool = False) -> None:
    """Creates a submission file from masks filenames.

    Args:
        submission_filename (str): submission csv file path.
        masks_filenames (list): list of masks file paths.
        patch_size (int, optional): patch size. Defaults to 16.
        foreground_threshold (float, optional): foreground_threshold.
        Defaults to 0.25.
        clean (bool, optional): clean the patches by a neighbor method.
        Defaults to False.
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in tqdm(masks_filenames, desc='Create submission', unit='mask'):
            f.writelines(f'{s}\n' for s in mask_to_submission_strings(
                fn, patch_size, foreground_threshold=foreground_threshold,
                clean=clean))
