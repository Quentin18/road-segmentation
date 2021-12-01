"""
Submission script.
"""
import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Add root directory to path
from config import add_root_to_path
add_root_to_path()

# Imports from src
from src.datasets import SatelliteImagesDataset
from src.models import UNet
from src.path import (DATA_TEST_IMG_PATH, DEFAULT_SUBMISSION_MASK_DIR,
                      DEFAULT_WEIGHTS_PATH, OUT_DIR, create_dirs,
                      extract_archives)
from src.predicter import Predicter
from src.submission import masks_to_submission


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
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    # Define neural net
    model = UNet()
    model.to(device)

    # Load weights
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Create predicter
    predicter = Predicter(
        model=model,
        device=device,
        predictions_path=DEFAULT_SUBMISSION_MASK_DIR,
        data_loader=test_loader,
    )

    # Run prediction
    predicter.predict(args.predict_threshold)

    # CSV submission
    print("== Creation of mask images ==")
    masks_to_submission(
        submission_filename=os.path.join(OUT_DIR, 'unet_submission.csv'),
        masks_filenames=os.listdir(DEFAULT_SUBMISSION_MASK_DIR),
        foreground_threshold=args.submit_threshold,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predicting model for road segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='input batch size',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=608,
        help='target input image size',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=DEFAULT_WEIGHTS_PATH,
        help='output model path',
    )
    parser.add_argument(
        '--predict-threshold',
        type=float,
        default=0.2,
        help='threshold for predict',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed',
    )
    parser.add_argument(
        '--submit-threshold',
        type=float,
        default=0.2,
        help='threshold for submission',
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='number of workers for data loading',
    )
    args = parser.parse_args()
    main(args)
