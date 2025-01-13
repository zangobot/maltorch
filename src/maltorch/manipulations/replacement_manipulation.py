import torch
from torch import Tensor

from maltorch.initializers.initializers import ByteBasedInitializer
from maltorch.manipulations.byte_manipulation import ByteManipulation


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
