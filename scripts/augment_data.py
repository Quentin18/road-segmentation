"""
Script to create the augmented dataset.
"""
# Add root directory to path
from config import add_root_to_path
add_root_to_path()

# Imports from src
from src.data_augmentation import create_augmented_dataset
from src.path import DATA_TRAIN_AUG_PATH


def main() -> None:
    """
    Main to create the augmented dataset.
    """
    print('Create augmented dataset')
    create_augmented_dataset(replace=True)
    print('Dataset created at', DATA_TRAIN_AUG_PATH)


if __name__ == '__main__':
    main()
