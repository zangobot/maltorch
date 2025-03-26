"""PyTorch model trainer with early stopping and weighted BCE Loss."""

import torch.nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

class WeightedBCEPyTorchTrainer:
    """Trainer for PyTorch models with early stopping."""
    def __init__(self, optimizer: torch.optim.Optimizer, epochs: int = 5, scheduler: _LRScheduler = None, pos_weight: float = 1.0, neg_weight: float = 1.0) -> None:
        """
        Create PyTorch trainer.
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to use for training the model.
        epochs : int, optional
            Number of epochs, by default 5.
        scheduler : _LRScheduler, optional
            Scheduler for the optimizer, by default None.
        """
        self._epochs = epochs
        self._optimizer = optimizer
        self._scheduler = scheduler

        self.training_losses = []
        self.training_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []

        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def train(self, model: torch.nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              patience: int) -> torch.nn.Module:
        """
        Train model with given loaders and early stopping.
        Parameters
        ----------
        model : torch.nn.Module
            Pytorch model to be trained.
        train_loader : DataLoader
            Train data loader.
        val_loader : DataLoader
            Validation data loader.
        patience : int
            Number of epochs to wait before early stopping.
        Returns
        -------
        torch.nn.Module
            Trained model.
        """
        best_loss = float("inf")
        best_model = None
        patience_counter = 0
        for epoch in range(self._epochs):
            model = self.fit(model, train_loader)
            val_loss = self.validate(model, val_loader)
            if val_loss <= best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model)
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break
            print(
                f"Epoch {epoch}: val_loss = {val_loss}, best_loss = {best_loss}, patience_counter = {patience_counter}")
        return best_model

    def fit(self,
            model: torch.nn.Module,
            dataloader: DataLoader) -> torch.nn.Module:
        """
        Train model for one epoch with given loader.
        Parameters
        ----------
        model : torch.nn.Module
            Pytorch model to be trained.
        dataloader : DataLoader
            Train data loader.
        Returns
        -------
        torch.nn.Module
            Trained model.
        """
        device = next(model.parameters()).device
        model = model.train()
        running_loss = 0.0
        train_total = 0
        train_correct = 0
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            weights = torch.where(y == 1,
                              torch.tensor(self.pos_weight, device=device),
                              torch.tensor(self.neg_weight, device=device)).to(device)
            self._optimizer.zero_grad()
            outputs = model(x)
            outputs = outputs.squeeze()

            criterion = torch.nn.BCEWithLogitsLoss(weight=weights)
            loss = criterion(outputs, y.float())

            loss.backward()
            self._optimizer.step()

            running_loss += loss.item()
            y_preds = outputs.round()

            train_total += y.size(0)
            train_correct += (y_preds == y).sum().item()

        self.training_losses.append(running_loss / train_total)
        self.training_accuracies.append(train_correct / train_total)

        if self._scheduler is not None:
            self._scheduler.step()
        return model

    def validate(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader) -> float:
        """
        Validate model with given loader.
        Parameters
        ----------
        model : torch.nn.Module
            Pytorch model to be balidated.
        dataloader : DataLoader
            Validation data loader.
        Returns
        -------
        float
            Validation loss of the model.
        """
        running_loss = 0
        val_total = 0
        val_correct = 0
        device = next(model.parameters()).device
        model = model.eval()
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                x, y = x.to(device), y.to(device)
                weights = torch.where(y == 1,
                                      torch.tensor(self.pos_weight, device=device),
                                      torch.tensor(self.neg_weight, device=device)).to(device)
                outputs = model(x)
                outputs = outputs.squeeze()
                criterion = torch.nn.BCELoss(weight=weights)
                loss = criterion(outputs, y.float())

                running_loss += loss.item()
                y_preds = outputs.round()

                val_total += y.size(0)
                val_correct += (y_preds == y).sum().item()

            val_loss = running_loss / val_total
            self.validation_losses.append(running_loss / val_total)
            self.validation_accuracies.append(val_correct / val_total)
        return val_loss