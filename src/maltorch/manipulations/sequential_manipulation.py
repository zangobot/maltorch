import torch
from typing import List, Type
from maltorch.manipulations.byte_manipulation import ByteManipulation

class SequentialManipulation(ByteManipulation):
    """Apply a sequence of manipulations in order.

    Parameters
    ----------
    manipulation_types : list[type]
        List of classes that inherit from :class:`ByteManipulation`.
    """

    def __init__(self, manipulation_types: List[Type[ByteManipulation]]):
        # dummy lists for the base constructor – they will be overridden
        super().__init__(initializer=None, domain_constraints=[], perturbation_constraints=[])
        self.manipulation_types = manipulation_types
        self.num_manipulations = len(manipulation_types)
        # we keep a placeholder for the initializer; it will not be used
        self.initializer = None

    def initialize(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # No delta is needed – return an empty tensor
        zero_delta = torch.empty((0,), dtype=torch.long, device=samples.device)
        return samples, zero_delta

    def _apply_manipulation(self, x: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the manipulations indexed in *delta* to the single sample *x*.

        Each index refers to a class in ``self.manipulation_types``.  The
        manipulation is instantiated with default (empty) constraints and
        applied to *x* in order.
        """
        for idx in delta:
            # Instantiate the manipulation class with no constraints
            current_manipulation = self.manipulation_types[idx]
            manipulation = current_manipulation(initializer=None, domain_constraints=[], perturbation_constraints=[])
            # Apply the manipulation – its ``__call__`` handles constraints
            x, _ = manipulation(x, torch.empty(0, dtype=torch.long, device=x.device))
        return x, delta
