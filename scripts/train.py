"""
Traininig script.

Usage:
python3 train.py

To see the different options, run `python3 train.py --help`.
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
from src.models import SegNet, UNet
from src.path import (DATA_TRAIN_GT_PATH, DATA_TRAIN_IMG_PATH, create_dirs,
                      extract_archives, generate_log_filename,
                      generate_model_filename)
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

    # Define model
    if args.model == 'unet':
        model = UNet()
    elif args.model == 'segnet':
        model = SegNet()
    else:
        print(f'Error: unknown model {args.model}')
        return
    print('Model:', args.model)
    model.to(device)

    # Define a loss function and optimizer
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        weights_path=model_path,
        log_path=log_path,
        data_loader=train_loader,
        valid_data_loader=test_loader,
        notebook=args.notebook,
    )
    trainer.train(args.epochs)

    # Plot history
    trainer.history.plot(log_path.replace('.pickle', '.pdf'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training model for road segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='input batch size for training',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs to train',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=320,
        help='target input image size',
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
        default=1e-3,
        help='initial learning rate',
    )
    parser.add_argument(
        '--model',
        choices=('unet', 'segnet'),
        default='unet',
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
