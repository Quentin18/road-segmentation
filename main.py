import matplotlib.pyplot as plt

from src.datasets import SatelliteImagesTrainDataset
from src.path import DATA_TRAIN_PATH, extract_archives
from src.plot_utils import plot_image_mask


def main():
    extract_archives()
    dataset = SatelliteImagesTrainDataset(DATA_TRAIN_PATH)
    image, mask = dataset[10]
    plot_image_mask(image, mask)
    plt.show()


if __name__ == '__main__':
    main()
