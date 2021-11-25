"""
Predicting script.
"""
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets import SatelliteImagesTrainDataset, train_test_split
from src.nets import UNet
from src.path import (DATA_TRAIN_PATH, DEFAULT_PREDICTIONS_DIR,
                      DEFAULT_WEIGHTS_PATH, create_dirs, extract_archives)
from src.predicter import Predicter


def main(args: argparse.Namespace) -> None:
    """Main to predict.

    Args:
        args (argparse.Namespace): namespace of arguments.
    """
    print('== Start predicting ==')

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
    mask_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    # Define dataset
    dataset = SatelliteImagesTrainDataset(
        root_dir=DATA_TRAIN_PATH,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    print('Size of dataset:', len(dataset))

    image, mask = dataset[0]
    print('Image size:', image.shape)
    print('Mask size:', mask.shape)

    if args.split_ratio > 0:
        # Split train test
        train_set, test_set = train_test_split(
            dataset=dataset,
            test_ratio=args.split_ratio
        )
        print('Train size:', len(train_set))
        print('Test size:', len(test_set))

        # Define loader
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin_memory,
        )
    else:
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
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
        predictions_path=DEFAULT_PREDICTIONS_DIR,
        data_loader=test_loader,
    )

    # Run prediction
    predicter.predict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predicting U-Net model for road segmentation"
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
        default=400,
        help="target input image size (default: 400)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.2,
        help="train test split ratio. 0 to train the whole dataset "
             "(default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed (default: 0)",
    )
    args = parser.parse_args()
    main(args)
