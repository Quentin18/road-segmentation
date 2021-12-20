"""
Traininig script.

Usage:
python3 train.py

To see the different options, run `python3 train.py --help`.
"""
import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

# Add root directory to path
from config import add_root_to_path
add_root_to_path()

# Imports from src
from src.datasets import SatelliteImagesDataset, train_test_split
from src.loss import DiceLoss
from src.models import NestedUNet, SegNet, UNet
from src.path import (DATA_TRAIN_AUG_GT_PATH, DATA_TRAIN_AUG_IMG_PATH,
                      create_dirs, extract_archives, generate_log_filename,
                      generate_model_filename)
from src.trainer import Trainer

# Default config
MODEL = 'unet'
BATCH_SIZE = 10
EPOCHS = 100
LR = 1e-4
SEED = 0
SPLIT_RATIO = 0.2
WEIGHT_DECAY = 1e-4
WORKERS = 2

# Paths
IMG_DIR = DATA_TRAIN_AUG_IMG_PATH
GT_DIR = DATA_TRAIN_AUG_GT_PATH


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
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Define paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = generate_model_filename()
    print('Model file:', model_path)

    if args.log_path is not None:
        log_path = args.log_path
    else:
        log_path = generate_log_filename()
    print('Log file:', log_path)

    # Define dataset
    dataset = SatelliteImagesDataset(
        img_dir=IMG_DIR,
        gt_dir=GT_DIR,
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
            test_ratio=args.split_ratio,
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
            batch_size=1,
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

    # Define loss function and optimizer
    criterion = DiceLoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        patience=5,
        verbose=True,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        weights_path=model_path,
        log_path=log_path,
        data_loader=train_loader,
        valid_data_loader=test_loader,
        lr_scheduler=lr_scheduler,
        notebook=args.notebook,
    )
    trainer.train(args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training model for road segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='input batch size for training',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help='number of epochs to train',
    )
    parser.add_argument(
        '--log-path',
        type=str,
        default=None,
        help='output log path'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=LR,
        help='initial learning rate',
    )
    parser.add_argument(
        '--model',
        choices=('unet', 'segnet', 'nested-unet'),
        default=MODEL,
        help='model to use',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='output model path',
    )
    parser.add_argument(
        '--notebook',
        action='store_true',
        help='notebook mode',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='seed',
    )
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.2,
        help='train test split ratio. 0 to train the whole dataset',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='number of workers for data loading',
    )
    args = parser.parse_args()
    main(args)
