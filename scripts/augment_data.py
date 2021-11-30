"""
Script to create the augmented dataset.
"""
# Add root directory to path
from config import add_root_to_path
add_root_to_path()

# Imports from src
from src.data_augmentation import create_augmented_dataset


def main() -> None:
    """
    Main to create the augmented dataset.
    """
    create_augmented_dataset(replace=True)


if __name__ == '__main__':
    main()
