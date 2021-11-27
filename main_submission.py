"""
Submission script.
"""
import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets import SatelliteImagesDataset
from src.nets import UNet
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
    pin_memory = device == 'cuda'

    # Define transforms
    image_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Define dataset
    dataTestSet = SatelliteImagesDataset(
        img_dir=DATA_TEST_IMG_PATH,
        image_transform=image_transform)
    print('Size of dataset:', len(dataTestSet))

    image, _ = dataTestSet[0]
    print('Image size:', image.shape)

    test_loader = DataLoader(
        dataset=dataTestSet,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    # Define neural net
    model = UNet()
    model.to(device)

    # Load weights
    state_dict = torch.load(args.weights_path, map_location=device)
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
    mask_path_submission = os.path.join(OUT_DIR, 'submission')
    mask_filename = os.listdir(mask_path_submission)
    masks_to_submission(os.path.join(OUT_DIR, 'UNet_submission.csv'),
                        mask_filename,
                        foreground_threshold=args.proba_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predicting U-Net model for road segmentation"
    )
    parser.add_argument(
        "--predict_threshold",
        type=float,
        default=0.6,
        help="threshold for predicct (default: 0.6)",
    )
    parser.add_argument(
        "--proba_threshold",
        type=float,
        default=0.25,
        help="threshold for submission (default: 0.25)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="input batch size for training (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=DEFAULT_WEIGHTS_PATH,
        help="output weights path"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=608,
        help="target input image size (default: 608)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed (default: 0)",
    )
    args = parser.parse_args()
    main(args)
