"""
Predicting script.
"""
import argparse
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

# Add root directory to path
from config import add_root_to_path
add_root_to_path()

# Imports from src
from src.data_augmentation import AUG_IMG_SIZE
from src.datasets import SatelliteImagesDataset, train_test_split
from src.models import SegNet, UNet
from src.path import (DATA_TRAIN_AUG_GT_PATH, DATA_TRAIN_AUG_IMG_PATH,
                      DEFAULT_PREDICTIONS_DIR, DEFAULT_WEIGHTS_PATH,
                      create_dirs, extract_archives)
from src.plot_utils import plot_images
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
        train_set, test_set = train_test_split(
            dataset=dataset,
            test_ratio=args.split_ratio,
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
    )

    # Run prediction
    accuracy, f1 = predicter.predict(proba_threshold=0.25)
    print('Accuracy:', accuracy)
    print('F1 score:', f1)

    # Plot image-mask-prediction
    # TODO to improve
    print('Save img-mask-pred')
    if args.split_ratio > 0:
        for (image, mask), pred_path in zip(
            test_set, predicter.predictions_filenames
        ):
            filename = os.path.basename(pred_path).replace(
                'prediction', 'img_mask_pred'
            )
            pred = Image.open(pred_path)
            path = os.path.join(os.path.dirname(pred_path), filename)
            plot_images(image, mask, pred, path=path)
    print('End img-mask-pred')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predicting for road segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='input batch size for predicting',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=AUG_IMG_SIZE,
        help='target input image size',
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
        default=DEFAULT_WEIGHTS_PATH,
        help='output model path',
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
