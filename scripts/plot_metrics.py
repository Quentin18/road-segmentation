"""
Plotting metrics script.
"""
import argparse

import matplotlib.pyplot as plt

# Add root directory to path
from config import add_root_to_path
add_root_to_path()

# Imports from src
from src.path import DEFAULT_LOSSES_PATH
from src.trainer import History


def main(args: argparse.Namespace) -> None:
    """Main to plot metrics.

    Args:
        args (argparse.Namespace): namespace of arguments.
    """
    history = History()
    history.load(args.file)
    history.plot()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plotting metrics from training"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=DEFAULT_LOSSES_PATH,
        help="pickle file",
    )
    args = parser.parse_args()
    main(args)
