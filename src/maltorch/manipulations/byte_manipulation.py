import abc

import torch
from secmlt.manipulations.manipulation import Manipulation
from secmlt.optimization.constraints import Constraint

from maltorch.initializers.initializers import ByteBasedInitializer


class ByteManipulation(Manipulation, abc.ABC):
    def __init__(
        self,
        initializer: ByteBasedInitializer,
        domain_constraints: list[Constraint],
        perturbation_constraints: list[Constraint],
    ):
        super().__init__(domain_constraints, perturbation_constraints)
        self.initializer = initializer
        self.indexes_to_perturb = []

    @abc.abstractmethod
    def initialize(self, samples: torch.Tensor) -> [torch.Tensor, torch.Tensor]: ...
