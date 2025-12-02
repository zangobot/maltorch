import torch
from torch import Tensor

from maltorch.initializers.initializers import ByteBasedInitializer, IdentityInitializer
from maltorch.manipulations.byte_manipulation import ByteManipulation


class GradientFreeWrapperManipulation(ByteManipulation):
    def __init__(
        self,
        inner_manipulation: ByteManipulation,
        domain_constraints=None,
        perturbation_constraints=None,
    ):
        if domain_constraints is None:
            domain_constraints = []
        if perturbation_constraints is None:
            perturbation_constraints = []
        self.inner_manipulation = inner_manipulation
        super().__init__(IdentityInitializer(), domain_constraints, perturbation_constraints)

    def _apply_manipulation(
        self, x: torch.Tensor, delta: torch.Tensor
    ) -> tuple[Tensor, Tensor]:
        x, scaled_delta = self.inner_manipulation(x, delta * 255)
        return x, scaled_delta / 255

    def initialize(self, samples: torch.Tensor):
        self.indexes_to_perturb = []
        modified_sample, delta, indexes = self.inner_manipulation.initializer(samples.data)
        self.indexes_to_perturb = indexes
        samples.data = modified_sample
        return samples, delta / 255
