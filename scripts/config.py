"""
Configuration helpers.
"""
import os
import sys

# Paths
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(DIRNAME)


def add_root_to_path() -> None:
    """
    Adds the root of the repository to the path, so that we can import
    functions from src.
    """
    sys.path.append(ROOT_DIR)
