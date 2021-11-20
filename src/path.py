"""
Paths and archives management.
"""
import os
import zipfile


# Directories paths
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(DIRNAME)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
OUT_DIR = os.path.join(ROOT_DIR, 'out')

# Dataset paths
DATA_TEST_PATH = os.path.join(DATA_DIR, 'test_set_images')
DATA_TRAIN_PATH = os.path.join(DATA_DIR, 'training')


def extract_archives() -> None:
    """Extracts the archives in the data directory if needed."""
    for path in (DATA_TEST_PATH, DATA_TRAIN_PATH):
        if not os.path.exists(path):
            zip_filename = path + '.zip'
            with zipfile.ZipFile(zip_filename, 'r') as zf:
                zf.extractall(DATA_DIR)


def create_out_dir() -> None:
    """Creates the `out` directory if needed."""
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
