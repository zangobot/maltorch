import torch
from secml2.manipulations.manipulation import Manipulation
from secml2.optimization.initializer import Initializer


class ReplacementManipulation(Manipulation):
    def __init__(self, indexes_to_perturb: torch.Tensor):
        self.indexes_to_perturb = indexes_to_perturb

    def __call__(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x[:, self.indexes_to_perturb] = delta
        return x

    def invert(self, x: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        delta = x_adv[:, self.indexes_to_perturb].data
        return delta


class ReplacementInitializer(Initializer):
    ...
