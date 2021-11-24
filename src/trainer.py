"""
Neural network trainer class.
"""
import pickle
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


class Trainer:
    """
    Trainer class to train a neural network.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: str,
        weights_path: str,
        log_path: str,
        data_loader: DataLoader,
        valid_data_loader: DataLoader = None,
        save_period: int = 5,
    ) -> None:
        """Inits the trainer.

        Args:
            model (torch.nn.Module): neural network model.
            criterion (torch.nn.modules.loss._Loss): loss function.
            optimizer (torch.optim.Optimizer): optimizer function.
            epochs (int): number of epochs.
            device (str): device (cpu or cuda).
            weights_path (str): path to save weights for the model.
            log_path (str): path to save training history.
            data_loader (DataLoader): loader of train data.
            valid_data_loader (DataLoader, optional): loader of test data.
            Defaults to None.
            save_period (int, optional): period to save weights and history.
            Defaults to 5.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.weights_path = weights_path
        self.log_path = log_path
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.save_period = save_period
        self.do_validation = self.valid_data_loader is not None

        # Calculate steps per epoch for train and valid set
        self.train_steps = len(self.data_loader.dataset) // \
            self.data_loader.batch_size
        if self.do_validation:
            self.valid_steps = len(self.valid_data_loader.dataset) // \
                self.valid_data_loader.batch_size

        # Initialize a dictionary to store training history
        self.history = {
            'train_loss': list(),
            'test_loss': list(),
        }

    def _train_epoch(self, epoch: int) -> float:
        """Training for an epoch.

        Args:
            epoch (int): number of the epoch.

        Returns:
            float: train loss.
        """
        # Set the model in training mode
        self.model.train()

        # Initialize the total training loss
        total_train_loss = 0

        # Loop over the training set
        with tqdm(
            self.data_loader,
            desc=f'Train epoch {epoch}',
            unit='batch',
            leave=False,
        ) as t:
            t.set_postfix(loss=None)
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

                # Add the loss to the total training loss
                t.set_postfix(loss=loss.item())
                total_train_loss += loss.item()

        # Calculate the average training loss
        train_loss = total_train_loss / self.train_steps

        return train_loss

    def _valid_epoch(self, epoch: int) -> float:
        """Validating for an epoch.

        Args:
            epoch (int): number of the epoch.

        Returns:
            float: test loss.
        """
        # Set the model in evaluation mode
        self.model.eval()

        # Initialize the total validation loss
        total_test_loss = 0

        # Switch off autograd
        with torch.no_grad():
            # Loop over the validation set
            with tqdm(
                self.valid_data_loader,
                desc=f'Valid epoch {epoch}',
                unit='batch',
                leave=False,
            ) as t:
                t.set_postfix(loss=None)
                for data, target in t:
                    # Send the input to the device
                    data, target = data.to(self.device), target.to(self.device)

                    # Make the predictions and calculate the validation loss
                    output = self.model(data)
                    loss = self.criterion(output, target)

                    # Add the loss to the total validation loss
                    t.set_postfix(loss=loss.item())
                    total_test_loss += loss.item()

        # Calculate the average validation loss
        test_loss = total_test_loss / self.train_steps

        return test_loss

    def _save_model(self) -> None:
        """
        Saves the weights and the history.
        """
        # Save weights
        weights = self.model.state_dict()
        torch.save(weights, self.weights_path)

        # Save history
        with open(self.log_path, 'wb') as f:
            pickle.dump(self.history, f)

    def train(self) -> None:
        """
        Trains the model.
        """
        print('Start training.')
        t_start = time.time()

        train_loss, test_loss = None, None

        # Loop over epochs
        with trange(1, self.epochs + 1, desc='Training', unit='epoch') as t:
            t.set_postfix(train_loss=train_loss, test_loss=test_loss)
            for epoch in t:
                # Train
                train_loss = self._train_epoch(epoch)
                self.history['train_loss'].append(train_loss)
                t.set_postfix(train_loss=train_loss, test_loss=test_loss)

                if self.do_validation:
                    # Valid
                    test_loss = self._valid_epoch(epoch)
                    self.history['test_loss'].append(test_loss)
                    t.set_postfix(train_loss=train_loss, test_loss=test_loss)

                # Save model
                if epoch % self.save_period == 0:
                    self._save_model()

        t_end = time.time()
        print(f'End training. Time: {t_end - t_start:.3f}s.')
