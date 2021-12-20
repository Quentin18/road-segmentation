"""
Predicting script.

Usage:
python3 predict.py

To see the different options, run `python3 predict.py --help`.
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
from src.datasets import SatelliteImagesDataset, train_test_split
from src.models import NestedUNet, SegNet, UNet
from src.path import (DATA_TRAIN_AUG_GT_PATH, DATA_TRAIN_AUG_IMG_PATH,
                      DEFAULT_PREDICTIONS_DIR, MODELS_DIR, create_dirs,
                      extract_archives)
from src.predicter import Predicter

# Default config
DEFAULT_MODEL = 'unet'
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'best-model-unet.pt')
PROBA_THRESHOLD = 0.25
CLEAN = True


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
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Define dataset
    dataset = SatelliteImagesDataset(
        img_dir=DATA_TRAIN_AUG_IMG_PATH,
        gt_dir=DATA_TRAIN_AUG_GT_PATH,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    print('Size of dataset:', len(dataset))

    image, mask = dataset[0]
    print('Image size:', image.shape)
    print('Mask size:', mask.shape)

    if args.split_ratio > 0:
        # Split train test
        _, test_set = train_test_split(
            dataset=dataset,
            test_ratio=args.split_ratio,
            seed=args.seed,
        )
    else:
        test_set = dataset

    print('Test size:', len(test_set))

    # Define loader
    test_loader = DataLoader(
        dataset=test_set,
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

    # Load model
    print('Load model:', args.model_path)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Create predicter
    predicter = Predicter(
        model=model,
        device=device,
        predictions_path=DEFAULT_PREDICTIONS_DIR,
        data_loader=test_loader,
        save_comparison=True,
    )

    # Run prediction
    accuracy, f1 = predicter.predict(
        proba_threshold=PROBA_THRESHOLD,
        clean=CLEAN,
    )
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 score: {f1:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predicting for road segmentation',
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
        '--split-ratio',
        type=float,
        default=0.2,
        help='train test split ratio. 0 to predict the whole dataset',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='number of workers for data loading',
    )
    args = parser.parse_args()
    main(args)
