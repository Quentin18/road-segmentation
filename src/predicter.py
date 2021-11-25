"""
Neural network predicter class.
"""
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.metrics import accuracy_score_tensors, f1_score_tensors


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
    ) -> None:
        """Inits the predicter.

        Args:
            model (torch.nn.Module): neural network model.
            device (str): device (cpu or cuda).
            predictions_path (str): path to the directory of predictions.
            data_loader (DataLoader): data loader to predict.
        """
        self.model = model
        self.device = device
        self.predictions_path = predictions_path
        self.data_loader = data_loader

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

        # Switch off autograd
        with torch.no_grad():
            # Loop over the dataset
            with tqdm(self.data_loader, unit='batch') as t:
                for i, (data, target) in enumerate(t):
                    filename = f'prediction_{i + 1:03d}.png'
                    t.set_description(desc=filename)

                    # Send the input to the device
                    data, target = data.to(self.device), target.to(self.device)

                    # Make the predictions
                    output = self.model(data)

                    # Get labels
                    output = (output > proba_threshold)

                    # Compute metrics
                    if target is not None:
                        accuracy = accuracy_score_tensors(target, output)
                        f1 = f1_score_tensors(target, output)
                        t.set_postfix(acuracy=accuracy, f1=f1)
                        accuracy_scores.append(accuracy)
                        f1_scores.append(f1)

                    # Save mask
                    output = (output * 255).type(torch.float32)
                    output_path = os.path.join(self.predictions_path, filename)
                    save_image(output, output_path)

        # Compute average metrics
        if accuracy_scores and f1_scores:
            avg_accuracy = sum(accuracy_scores).item() / len(accuracy_scores)
            avg_f1 = sum(f1_scores).item() / len(f1_scores)
            print(f'Accuracy: {100 * avg_accuracy:.3f}%')
            print(f'F1 score: {100 * avg_f1:.3f}%')

        return avg_accuracy, avg_f1


class PredicterSubmission:
    """
    Predicter class.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        predictions_path: str,
        data_loader: DataLoader,
    ) -> None:
        """Inits the predicter.

        Args:
            model (torch.nn.Module): neural network model.
            device (str): device (cpu or cuda).
            predictions_path (str): path to the directory of predictions.
            data_loader (DataLoader): data loader to predict.
        """
        self.model = model
        self.device = device
        self.predictions_path = predictions_path
        self.data_loader = data_loader

    def predict(self, proba_threshold: float = 0.5) -> None:
        """Predicts the masks of images.

        Args:
            proba_threshold (float): probability threshold.
        """
        # Set the model in evaluation mode
        self.model.eval()

        # Switch off autograd
        with torch.no_grad():
            # Loop over the dataset
            for i, (data) in enumerate(
                tqdm(self.data_loader, unit='batch')
            ):
                # Send the input to the device
                data = data.to(self.device)

                # Make the predictions
                output = self.model(data)

                # Save mask
                output_path = os.path.join(
                    self.predictions_path, f'prediction_{i + 1:03d}.png'
                )
                output = (output > proba_threshold) * 255
                output = output.type(torch.float32)
                save_image(output, output_path)
