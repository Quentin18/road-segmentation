"""
Traininig script.
"""
import argparse

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

# Add root directory to path
from config import add_root_to_path
add_root_to_path()

# Imports from src
from src.datasets import SatelliteImagesDataset, train_test_split
from src.nets import UNet
# from src.SegNet import SegNet
from src.path import (DATA_TRAIN_GT_PATH, DATA_TRAIN_IMG_PATH,
                      DEFAULT_LOSSES_PATH, DEFAULT_WEIGHTS_PATH, create_dirs,
                      extract_archives)
from src.plot_utils import plot_loss
from src.trainer import Trainer


def main(args: argparse.Namespace):
    """Main to train.

    Args:
        args (argparse.Namespace): namespace of arguments.
    """
    print('== Start training ==')
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
    dataset = SatelliteImagesDataset(
        img_dir=DATA_TRAIN_IMG_PATH,
        gt_dir=DATA_TRAIN_GT_PATH,
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

        # Define loaders
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=pin_memory,
        )
        test_loader = None

    # Define neural net
    model = UNet()
    # model = SegNet()
    model.to(device)

    # Define a loss function and optimizer
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        weights_path=args.weights_path,
        log_path=args.log_path,
        data_loader=train_loader,
        valid_data_loader=test_loader,
    )
    trainer.train()

    # Plot train test loss
    # TODO to improve
    plot_loss(
        train_loss=trainer.history['train_loss'],
        test_loss=trainer.history['test_loss'],
        path=DEFAULT_LOSSES_PATH.replace('.pickle', '.png'),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for road segmentation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="input batch size for training (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="initial learning rate (default: 1e-3)",
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
        "--log-path",
        type=str,
        default=DEFAULT_LOSSES_PATH,
        help="output log path"
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
