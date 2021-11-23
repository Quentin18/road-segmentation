"""
Main script.

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
https://github.com/mateuszbuda/brain-segmentation-pytorch
https://www.pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
"""
import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import trange

from src.datasets import SatelliteImagesTrainDataset, train_test_split
from src.nets import UNet
from src.path import DATA_TRAIN_PATH, WEIGHTS_DIR, extract_archives
from src.plot_utils import plot_loss

TRAIN = False
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'weights.pt')
LOSSES_PATH = os.path.join(WEIGHTS_DIR, 'losses.pickle')


def main(args: argparse.Namespace):
    # Extract archives if needed
    extract_archives()

    # Set seed
    torch.manual_seed(0)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else args.device)

    # Define transforms
    image_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor()
    ])

    # Define dataset
    dataset = SatelliteImagesTrainDataset(
        root_dir=args.data_dir,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    image, mask = dataset[0]
    print('Image size:', image.shape)
    print('Mask size:', mask.shape)

    # Split train test
    train_set, test_set = train_test_split(dataset, test_ratio=0.2)
    print('Train size:', len(train_set))
    print('Test size:', len(test_set))

    # Define loaders
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    # Define neural net
    net = UNet()
    net.to(device)

    # Define a loss function and optimizer
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(net.parameters(), lr=args.lr)

    # Calculate steps per epoch for training and test set
    train_steps = len(train_set) // args.batch_size
    test_steps = len(test_set) // args.batch_size

    # Initialize a dictionary to store training history
    history = {"train_loss": list(), "test_loss": list()}

    # Train the network
    if TRAIN:
        print("[INFO] training the network...")
        t_start = time.time()

        # Loop over epochs
        for epoch in trange(args.epochs, unit='epoch'):
            # Set the model in training mode
            net.train()

            # initialize the total training and validation loss
            total_train_toss = total_test_loss = 0

            # Loop over the training set
            for i, data in enumerate(train_loader):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # Send the input to the device
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Add the loss to the total training loss
                total_train_toss += loss

            # Switch off autograd
            with torch.no_grad():
                # Set the model in evaluation mode
                net.eval()

                # loop over the Validation set
                for data in test_loader:
                    # Get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    # Send the input to the device
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Make the predictions and calculate the validation loss
                    pred = net(inputs)
                    loss = criterion(pred, labels)
                    total_test_loss += loss

                # Calculate the average training and validation loss
                avg_train_loss = total_train_toss / train_steps
                avg_test_loss = total_test_loss / test_steps

                # Update our training history
                history["train_loss"].append(
                    avg_train_loss.cpu().detach().numpy()
                )
                history["test_loss"].append(
                    avg_test_loss.cpu().detach().numpy())

                # Print the model training and validation information
                print("[INFO] EPOCH: {}/{}".format(epoch + 1, args.epochs))
                print("Train loss: {:.6f}, Test loss: {:.4f}".format(
                    avg_train_loss, avg_test_loss))

        # Display the total time needed to perform the training
        t_end = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(
            t_end - t_start))

        # Save model
        torch.save(net.state_dict(), args.weights)

        # Saves the losses and accuracies
        with open(LOSSES_PATH, 'wb') as f:
            pickle.dump(history, f)

    # Load losses
    with open(LOSSES_PATH, 'rb') as f:
        data = pickle.load(f)
    train_loss, test_loss = data['train_loss'], data['test_loss']

    plot_loss(train_loss, test_loss)
    plt.show()

    # Test the network on the test data
    net.load_state_dict(torch.load(args.weights))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Training U-Net model for road segmentation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
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
        "--device",
        type=str,
        default="cpu",
        help="device for training (default: cpu)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=WEIGHTS_PATH,
        help="output weights path"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_TRAIN_PATH,
        help="dataset folder path"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=400,
        help="target input image size (default: 400)",
    )
    args = parser.parse_args()
    main(args)
