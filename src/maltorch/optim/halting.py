import torch
from abc import abstractmethod, ABC


class Halting(ABC):
    @abstractmethod
    def halt_loop(self, loss):
        ...


class EarlyStopping(Halting):

    def __init__(self, patience=5):
        self.patience = patience
        self.current_patience = patience
        self.min_loss = torch.inf

    def halt_loop(self, loss):
        current_loss = loss.detach().cpu()
        if loss.shape == ():
            current_loss = current_loss.item()
        else:
            current_loss = current_loss.mean().item()
        if self.min_loss > current_loss:
            self.min_loss = current_loss
            self.current_patience = self.patience
        else:
            self.current_patience -= 1
        if self.current_patience <= 0:
            self.current_patience = self.patience
            self.min_loss = torch.inf
            return True
        return False


