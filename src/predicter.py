"""
Neural network predicter class.
"""
import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks, save_image
from tqdm import tqdm


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
            for i, (data, target) in enumerate(
                tqdm(self.data_loader, unit='batch')
            ):
                # Send the input to the device
                data, target = data.to(self.device), target.to(self.device)

                # Make the predictions
                output = self.model(data)

                # Save mask
                output_path = os.path.join(
                    self.predictions_path, f'prediction_{i + 1:03d}.png'
                )
                output = (output > proba_threshold) * 255
                output = output.type(torch.float32)
                save_image(output, output_path)
