"""
Main script.

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import os

# import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets import SatelliteImagesTrainDataset, train_test_split
from src.nets import UNet
from src.path import DATA_TRAIN_PATH, MODELS_DIR, extract_archives
# from src.plot_utils import plot_image_mask

NUM_CHANNELS = 3
IMG_WIDTH = 32
IMG_HEIGHT = 32
BATCH_SIZE = 16
NUM_EPOCHS = 100
MODEL_PATH = os.path.join(MODELS_DIR, 'convnet_model.pth')


def main():
    # Extract archives if needed
    extract_archives()

    # Define transforms
    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor()
    ])

    # Define dataset
    dataset = SatelliteImagesTrainDataset(
        root_dir=DATA_TRAIN_PATH,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    image, mask = dataset[0]
    print('Image size:', image.shape)
    print('Mask size:', mask.shape)

    # Split train test
    trainset, testset = train_test_split(dataset, 0.2)
    print('Train size:', len(trainset))
    print('Test size:', len(testset))

    # Define loaders
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    # Define neural net
    net = UNet()  # TODO to implement

    # Define a loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # Save model
    torch.save(net.state_dict(), MODEL_PATH)

    # Test the network on the test data
    # TODO

    # image, mask = dataset[10]
    # plot_image_mask(image, mask)
    # plt.show()


if __name__ == '__main__':
    main()
