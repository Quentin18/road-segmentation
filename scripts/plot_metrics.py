"""
Plotting metrics script.

Usage:
python3 plot_metrics.py --file FILE

`FILE` must be a `pickle` file.
"""
import argparse

import matplotlib.pyplot as plt

# Add root directory to path
from config import add_root_to_path
add_root_to_path()

# Imports from src
from src.trainer import History


def main(args: argparse.Namespace) -> None:
    """Main to plot metrics.

    Args:
        args (argparse.Namespace): namespace of arguments.
    """
    history = History()
    history.load(path=args.file)
    history.plot(path=args.save_path)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plotting metrics from training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--file',
        type=str,
        help='pickle file',
        required=True,
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default=None,
        help='path to save the figure',
    )
    args = parser.parse_args()
    main(args)
