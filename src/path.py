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

# Training paths
DEFAULT_LOSSES_PATH = os.path.join(MODELS_DIR, 'losses.pickle')
DEFAULT_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'weights.pt')
DEFAULT_PARAMETERS_PATH = os.path.join(MODELS_DIR, 'parameters.pickle')

# Testing paths
DEFAULT_PREDICTIONS_DIR = os.path.join(OUT_DIR, 'predictions')
DEFAULT_SUBMISSION_MASK_DIR = os.path.join(OUT_DIR, 'submission')


def extract_archives() -> None:
    """Extracts the archives in the data directory if needed."""
    for path in (DATA_TEST_PATH, DATA_TRAIN_PATH):
        if not os.path.exists(path):
            zip_filename = path + '.zip'
            with zipfile.ZipFile(zip_filename, 'r') as zf:
                zf.extractall(DATA_DIR)


def create_dirs() -> None:
    """Creates directories if needed."""
    for path in (MODELS_DIR, OUT_DIR, DEFAULT_PREDICTIONS_DIR):
        if not os.path.exists(path):
            os.mkdir(path)
