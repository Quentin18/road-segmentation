import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets import SatelliteImagesDataset
from src.nets import UNet
from src.path import (DATA_TRAIN_GT_PATH, DATA_TRAIN_IMG_PATH,
                      DEFAULT_PARAMETERS_PATH, DEFAULT_PREDICTIONS_DIR,
                      DEFAULT_WEIGHTS_PATH, create_dirs, extract_archives)
from src.plot_utils import plot_validation_F1
from src.predicter import Predicter


def main(args: argparse.Namespace) -> None:
    """Main to do validation on threshold parameter on F1 score

    Args:
        args (argparse.Namespace): namespace of arguments.
    """
    print('== Start Validation on threshold on F1 score==')

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

    # Define loaders
    train_loader = DataLoader(
        dataset=dataset,
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
        predictions_path=DEFAULT_PREDICTIONS_DIR,
        data_loader=train_loader,
    )

    # Run prediction
    f1_score_list = list()
    for threshold in args.threshold_validation:
        _, f1 = predicter.predict(proba_threshold=threshold)
        f1_score_list.append(f1)

    optimum_threshold = args.threshold_validation[np.argmax(f1_score_list)]
    parameters = dict()
    parameters['threshold'] = optimum_threshold

    # Plot result
    path = os.path.join(os.path.dirname(DEFAULT_PARAMETERS_PATH),
                        'threshold.png')
    plot_validation_F1(f1_score_list, args.threshold_validation, path=path)

    # Save optimum
    with open(DEFAULT_PARAMETERS_PATH, 'wb') as f:
        pickle.dump(parameters, f)


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
    parser.add_argument(
        "--threshold-validation",
        type=list,
        default=np.linspace(0, 1, 10),
        help="threshold to test for validation"
    )
    args = parser.parse_args()
    main(args)
