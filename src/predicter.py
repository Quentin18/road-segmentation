"""
Neural network predicter class.
"""
import os
import shutil
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook

from src.metrics import accuracy_score_tensors, f1_score_tensors
from src.plot_utils import plot_images
from src.submission import masks_to_submission


class Predicter:
    """
    Predicter class.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        predictions_path: str,
        data_loader: DataLoader,
        save_comparison: bool = True,
        notebook: bool = False,
    ) -> None:
        """Inits the predicter.

        Args:
            model (torch.nn.Module): neural network model.
            device (str): device (cpu or cuda).
            predictions_path (str): path to the directory of predictions.
            data_loader (DataLoader): data loader to predict.
            save_comparison (bool, optional): True to save comparison between
            image, groundtruth and prediction. Defaults to False.
            notebook (bool, optional): True if predicting is done in a notebook
            (to display progress bar properly). Defaults to False.
        """
        assert data_loader.batch_size == 1

        self.model = model
        self.device = device
        self.predictions_path = predictions_path
        self.data_loader = data_loader
        self.nb_imgs = len(data_loader)
        self.predictions_filenames = list()

        # Extra directory containing comparisons with images and groundtruth
        if save_comparison:
            self.extra_path = os.path.join(self.predictions_path, 'extra')
        else:
            self.extra_path = None

        # Set progress bar functions
        self.tqdm = tqdm_notebook if notebook else tqdm

        self._init_directories()

    def _init_directories(self) -> None:
        """Inits the output directories.
        """
        # Reset predictions directory
        shutil.rmtree(self.predictions_path, ignore_errors=True)

        # Create predictions directory
        os.makedirs(self.predictions_path, exist_ok=True)

        # Create extra directory
        if self.extra_path is not None:
            os.makedirs(self.extra_path, exist_ok=True)

    def _get_pred_filename(self, index: int) -> str:
        """Returns the filename of the prediction.

        Args:
            index (int): index of the image in the dataset.

        Returns:
            str: filename of the prediction.
        """
        if self.nb_imgs > 1000:
            return f'prediction_{index + 1:04d}.png'
        return f'prediction_{index + 1:03d}.png'

    def _predict_labels(
        self,
        output: torch.Tensor,
        proba_threshold: float,
    ) -> torch.Tensor:
        """Predicts the labels for an output.

        Args:
            output (torch.Tensor): tensor output.
            proba_threshold (float): probability threshold.

        Returns:
            torch.Tensor: tensor of 0 and 1.
        """
        return (output > proba_threshold).type(torch.uint8)

    def _save_mask(cls, output: torch.Tensor, filename: str) -> None:
        """Saves the mask as image.

        Args:
            output (torch.Tensor): tensor output.
            filename (str): filename.
        """
        array = torch.squeeze(output * 255).cpu().numpy()
        img = Image.fromarray(array)
        img.save(filename)

    def _save_comparison(
        self,
        index: int,
        data: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Saves the comparison image/grountruth/prediction.

        Args:
            index (int): index.
            data (torch.Tensor): image tensor.
            target (torch.Tensor): groundtruth image tensor.
            output (torch.Tensor): prediction tensor.
        """
        filename = os.path.join(
            self.extra_path, f'comparison_{index + 1:04d}.png'
        )
        data = torch.squeeze(data)
        target = torch.squeeze(target)
        output = torch.squeeze(output)
        plot_images(data, target, output, filename)

    def predict(self, proba_threshold: float = 0.25) -> Tuple[float, float]:
        """Predicts the masks of images.

        Args:
            proba_threshold (float): probability threshold.

        Returns:
            Tuple[float, float]: accuracy, f1 score.
        """
        # Set the model in evaluation mode
        self.model.eval()

        # Init metrics lists
        accuracy_scores, f1_scores = list(), list()
        avg_accuracy = avg_f1 = 0

        # Switch off autograd
        with torch.no_grad():
            # Loop over the dataset
            with self.tqdm(self.data_loader, unit='batch') as t:
                for i, (data, target) in enumerate(t):
                    filename = self._get_pred_filename(i)
                    t.set_description(desc=filename)

                    # Send the input to the device
                    data = data.to(self.device)
                    if target.dim() != 1:
                        target = target.to(self.device)

                    # Make the predictions
                    output = self.model(data)

                    # Get labels
                    output = self._predict_labels(output, proba_threshold)

                    # Compute metrics and save comparisons
                    if target.dim() != 1:
                        target = self._predict_labels(target, proba_threshold)
                        accuracy = accuracy_score_tensors(target, output)
                        f1 = f1_score_tensors(target, output)
                        t.set_postfix(acuracy=accuracy, f1=f1)
                        accuracy_scores.append(accuracy)
                        f1_scores.append(f1)
                        if self.save_comparison:
                            self._save_comparison(i, data, target, output)

                    # Save mask
                    output_path = os.path.join(self.predictions_path, filename)
                    self._save_mask(output, output_path)
                    self.predictions_filenames.append(output_path)

        # Compute average metrics
        if target.dim() != 1:
            avg_accuracy = sum(accuracy_scores).item() / len(accuracy_scores)
            avg_f1 = sum(f1_scores).item() / len(f1_scores)

        return avg_accuracy, avg_f1

    def create_submission(self, submission_filename: str) -> None:
        """Creates a submission file from the predictions.

        Args:
            submission_filename (str): submission csv file path.
        """
        masks_to_submission(submission_filename, self.predictions_filenames)
