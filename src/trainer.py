"""
Neural network trainer class.
"""
import pickle
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from tqdm.notebook import tnrange, tqdm_notebook

from src.metrics import accuracy_score_tensors, f1_score_tensors
from src.plot_utils import plot_history


class Trainer:
    """
    Trainer class to train a neural network.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        device: str,
        weights_path: str,
        log_path: str,
        data_loader: DataLoader,
        valid_data_loader: DataLoader = None,
        proba_threshold: float = 0.25,
        save_period: int = 5,
        early_stopping: bool = True,
        lr_scheduler: object = None,
        notebook: bool = False,
    ) -> None:
        """Inits the trainer.

        Args:
            model (torch.nn.Module): neural network model.
            criterion (torch.nn.modules.loss._Loss): loss function.
            optimizer (torch.optim.Optimizer): optimizer function.
            device (str): device (cpu or cuda).
            weights_path (str): path to save weights for the model.
            log_path (str): path to save training history.
            data_loader (DataLoader): loader of train data.
            valid_data_loader (DataLoader, optional): loader of valid data.
            Defaults to None.
            proba_threshold (float, optional): probability threshold to make
            predictions. Defaults to 0.25.
            save_period (int, optional): period to save weights and history.
            Defaults to 5.
            early_stopping (bool, optional): True to enable early stopping
            callback. Defaults to True.
            lr_scheduler (object, optional): learning rate scheduler.
            Defaults to None.
            notebook (bool, optional): True if training is done in a notebook
            (to display progress bar properly). Defaults to False.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.weights_path = weights_path
        self.log_path = log_path
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.proba_threshold = proba_threshold
        self.save_period = save_period
        self.do_validation = self.valid_data_loader is not None

        # Calculate steps per epoch for train and valid set
        self.train_steps = len(self.data_loader.dataset) // \
            self.data_loader.batch_size
        if self.do_validation:
            self.valid_steps = len(self.valid_data_loader.dataset) // \
                self.valid_data_loader.batch_size

        # Initialize the history
        self.history = History()

        # Initialize the EarlyStopping callback
        self.early_stopping = EarlyStopping() if early_stopping else None

        # Set progress bar functions
        self.tqdm = tqdm_notebook if notebook else tqdm
        self.trange = tnrange if notebook else trange

        # Learning rate scheduler
        self.lr_scheduler = lr_scheduler

    def _predict_labels(self, output: torch.Tensor) -> torch.Tensor:
        """Predicts the labels for an output.

        Args:
            output (torch.Tensor): tensor output.

        Returns:
            torch.Tensor: tensor of 0 and 1.
        """
        return (output > self.proba_threshold).type(torch.uint8)

    def _train_epoch(self, epoch: int) -> dict:
        """Training for an epoch.

        Args:
            epoch (int): number of the epoch.

        Returns:
            dict: training metrics.
        """
        # Set the model in training mode
        self.model.train()

        # Initialize metrics
        metrics = {
            'train_loss': 0.0,
            'train_accuracy': 0.0,
            'train_f1': 0.0,
        }

        # Loop over the training set
        with self.tqdm(
            self.data_loader,
            desc=f'Train epoch {epoch}',
            unit='batch',
            leave=False,
        ) as t:
            t.set_postfix(metrics)
            for data, target in t:
                # Send the input to the device
                data, target = data.to(self.device), target.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + backward + optimize
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                # Predict labels
                target = self._predict_labels(target)
                output = self._predict_labels(output)

                # Update metrics
                metrics['train_loss'] += loss.item()
                metrics['train_accuracy'] += accuracy_score_tensors(target,
                                                                    output)
                metrics['train_f1'] += f1_score_tensors(target, output)

                # Update progress bar
                t.set_postfix(metrics)

        # Average metrics
        for key in metrics:
            metrics[key] /= self.train_steps

        return metrics

    def _valid_epoch(self, epoch: int) -> dict:
        """Validating for an epoch.

        Args:
            epoch (int): number of the epoch.

        Returns:
            dict: validation metrics.
        """
        # Set the model in evaluation mode
        self.model.eval()

        # Initialize metrics
        metrics = {
            'valid_loss': 0.0,
            'valid_accuracy': 0.0,
            'valid_f1': 0.0,
        }

        # Switch off autograd
        with torch.no_grad():
            # Loop over the validation set
            with self.tqdm(
                self.valid_data_loader,
                desc=f'Valid epoch {epoch}',
                unit='batch',
                leave=False,
            ) as t:
                t.set_postfix(metrics)
                for data, target in t:
                    # Send the input to the device
                    data, target = data.to(self.device), target.to(self.device)

                    # Make the predictions and calculate the validation loss
                    output = self.model(data)
                    loss = self.criterion(output, target)

                    # Predict labels
                    target = self._predict_labels(target)
                    output = self._predict_labels(output)

                    # Update metrics
                    metrics['valid_loss'] += loss.item()
                    metrics['valid_accuracy'] += accuracy_score_tensors(target,
                                                                        output)
                    metrics['valid_f1'] += f1_score_tensors(target, output)

                    # Update progress bar
                    t.set_postfix(metrics)

        # Average metrics
        for key in metrics:
            metrics[key] /= self.valid_steps

        return metrics

    def _save_model(self) -> None:
        """
        Saves the weights and the history.
        """
        # Save weights
        weights = self.model.state_dict()
        torch.save(weights, self.weights_path)

        # Save history
        self.history.save(self.log_path)

    def train(self, epochs: int) -> None:
        """
        Trains the model.

        Args:
            epochs (int): number of epochs.
        """
        # Reset the history
        self.history.reset()

        print('Start training.')
        t_start = time.time()

        postfix = dict()

        # Init best f1
        best_f1 = 0.0

        # Loop over epochs
        with self.trange(1, epochs + 1, desc='Training', unit='epoch') as t:
            for epoch in t:
                # Train
                train_metrics = self._train_epoch(epoch)
                self.history.update(**train_metrics)
                postfix.update(train_metrics)
                f1 = train_metrics['train_f1']

                # Early stopping check
                train_loss = train_metrics['train_loss']
                if self.early_stopping and self.early_stopping(train_loss):
                    print(f'EarlyStopping: Stop training at epoch {epoch}.')
                    break

                if self.do_validation:
                    # Valid
                    valid_metrics = self._valid_epoch(epoch)
                    self.history.update(**valid_metrics)
                    postfix.update(valid_metrics)
                    f1 = valid_metrics['valid_f1']

                # Save model if better f1 score
                # (valid f1 if validation else train f1)
                if f1 > best_f1:
                    best_f1 = f1
                    postfix['best_f1'] = best_f1
                    self._save_model()

                # Adjust learning rate
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(train_loss)

                # Update progress bar
                t.set_postfix(postfix)

        t_end = time.time()
        print(f'End training. Time: {t_end - t_start:.3f}s.')


class History:
    """
    History handler to save the losses and other metrics during a training.
    """
    def __init__(self) -> None:
        self.epoch_metrics = dict()

    def reset(self) -> None:
        """
        Resets the history.
        """
        self.epoch_metrics.clear()

    def update(self, **kwargs) -> None:
        """
        Updates the history.
        """
        for key, value in kwargs.items():
            if key not in self.epoch_metrics:
                self.epoch_metrics[key] = [value]
            else:
                self.epoch_metrics[key].append(value)

    def save(self, path: str) -> None:
        """Saves the history in a pickle file.

        Args:
            path (str): path of the pickle file.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.epoch_metrics, f)

    def load(self, path: str) -> None:
        """Loads a history from a pickle file.

        Args:
            path (str): path of the pickle file.
        """
        with open(path, 'rb') as f:
            self.epoch_metrics = pickle.load(f)

    def plot(self, path: str = None) -> None:
        """
        Plots the history.

        Args:
            path(str, optional): path to save the figure. Defaults to None.
        """
        plot_history(self.epoch_metrics, path)


class EarlyStopping:
    """
    EarlyStopping handler can be used to stop the training if no improvement
    after a given number of events.
    """
    def __init__(self, min_delta: float = 0.0, patience: int = 5) -> None:
        """Inits the EarlyStopping handler.

        Args:
            min_delta (float, optional): a minimum loss decrease to qualify as
            an improvement. Defaults to 0.0.
            patience (int, optional): number of events to wait if no
            improvement and then stop the training. Defaults to 5.
        """
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = None

    def __call__(self, current_loss: float) -> bool:
        """Returns True if training needs to be stopping, False else.

        Args:
            current_loss (float): loss of the current epoch.

        Returns:
            bool: stop training.
        """
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 1
        else:
            if self.wait >= self.patience:
                return True
            self.wait += 1
        return False
