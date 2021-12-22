"""
Submission script.

Usage:
python3 run.py

To see the different options, run `python3 run.py --help`.

- The filename of the csv submission file is: `out/submission.csv`.
- The predicted masks (608x608) are saved in the `out/submission` directory.
- The masks with 16x16 patches (38x38) are saved in the `out/submission_patch`
directory.
- The overlays are saved in the `out/submission_overlay` directory.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Add root directory to path
from config import add_root_to_path
add_root_to_path()

# Imports from src
from src.datasets import SatelliteImagesDataset
from src.models import NestedUNet, SegNet, UNet
from src.path import (DATA_TEST_IMG_PATH, MODELS_DIR, OUT_DIR, create_dirs,
                      extract_archives)
from src.plot_utils import make_img_overlays
from src.predicter import Predicter
from src.submission import masks_to_submission, submission_to_masks

# Default config
DEFAULT_MODEL = 'unet'
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'best-model-unet.pt')
SUBMISSION_FILENAME = os.path.join(OUT_DIR, 'submission.csv')
SUBMISSION_DIRNAME = os.path.join(OUT_DIR, 'submission')
SUBMISSION_PATCH_DIRNAME = os.path.join(OUT_DIR, 'submission_patch')
SUBMISSION_OVERLAY_DIRNAME = os.path.join(OUT_DIR, 'submission_overlay')
PROBA_THRESHOLD = 0.2
FOREGROUND_THRESHOLD = 0.2
CLEAN = False


def main(args: argparse.Namespace) -> None:
    """Main to predict.

    Args:
        args (argparse.Namespace): namespace of arguments.
    """
    print('== Start predicting submission ==')

    # Extract archives and create directories if needed
    create_dirs()
    extract_archives()

    # Set seed
    torch.manual_seed(args.seed)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    pin_memory = device == 'cuda'

    # Define transforms
    image_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Define dataset
    train_set = SatelliteImagesDataset(
        img_dir=DATA_TEST_IMG_PATH,
        image_transform=image_transform,
    )
    print('Size of dataset:', len(train_set))

    image, _ = train_set[0]
    print('Image size:', image.shape)

    test_loader = DataLoader(
        dataset=train_set,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    # Define model
    if args.model == 'unet':
        model = UNet()
    elif args.model == 'segnet':
        model = SegNet()
    elif args.model == 'nested-unet':
        model = NestedUNet()
    else:
        print(f'Error: unknown model {args.model}')
        return
    print('Model:', args.model)
    model.to(device)

    # Load weights
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Create predicter
    predicter = Predicter(
        model=model,
        device=device,
        predictions_path=SUBMISSION_DIRNAME,
        data_loader=test_loader,
        save_comparison=False,
    )

    # Run prediction
    predicter.predict(proba_threshold=PROBA_THRESHOLD, clean=CLEAN)

    # CSV submission
    print("== Creation of mask images ==")
    masks_to_submission(
        submission_filename=SUBMISSION_FILENAME,
        masks_filenames=predicter.predictions_filenames,
        foreground_threshold=FOREGROUND_THRESHOLD,
    )

    # Create back masks to check submission
    print("== Creation of patch mask images ==")
    submission_to_masks(
        submission_filename=SUBMISSION_FILENAME,
        masks_dirname=SUBMISSION_PATCH_DIRNAME,
    )

    # Create overlays
    print("== Creation of overlays ==")
    make_img_overlays(
        image_dir=DATA_TEST_IMG_PATH,
        mask_dir=SUBMISSION_PATCH_DIRNAME,
        output_dir=SUBMISSION_OVERLAY_DIRNAME,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predicting model for road segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model',
        choices=('unet', 'segnet', 'nested-unet'),
        default=DEFAULT_MODEL,
        help='model to use',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help='model path',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='number of workers for data loading',
    )
    args = parser.parse_args()
    main(args)
