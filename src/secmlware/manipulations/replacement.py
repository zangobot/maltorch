import abc

import torch
from secmlt.manipulations.manipulation import Manipulation
from secmlt.optimization.constraints import Constraint
from torch import Tensor

from secmlware.initializers.initializers import ByteBasedInitializer


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


class ReplacementManipulation(ByteManipulation):
    def __init__(
        self,
        initializer: ByteBasedInitializer,
        domain_constraints=None,
        perturbation_constraints=None,
    ):
        if domain_constraints is None:
            domain_constraints = []
        if perturbation_constraints is None:
            perturbation_constraints = []
        super().__init__(initializer, domain_constraints, perturbation_constraints)

    def _apply_manipulation(
        self, x: torch.Tensor, delta: torch.Tensor
    ) -> tuple[Tensor, Tensor]:
        for i, idx in enumerate(self.indexes_to_perturb):
            x[i, idx[idx != -1]] = delta[i].long()
        return x, delta

    def initialize(self, samples: torch.Tensor):
        self.indexes_to_perturb = []
        samples.data, delta, indexes = self.initializer(samples.data)
        self.indexes_to_perturb = indexes
        return samples, delta
