import torch

from maltorch.initializers.initializers import ByteBasedInitializer
from maltorch.utils.pe_operations import content_shift_manipulation


class ContentShiftInitializer(ByteBasedInitializer):
    def __init__(self, preferred_manipulation_size, random_init=False):
        super().__init__(random_init=random_init)
        self.manipulation_size = preferred_manipulation_size

    def __call__(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        X = []
        deltas = []
        indexes = []
        for x_i in x:
            x_i, shift_indexes = content_shift_manipulation(x_i, self.manipulation_size)
            X.append(x_i)
            delta = (
                torch.zeros(len(shift_indexes))
                if not self.random_init
                else torch.randint(0, 255, (len(shift_indexes),))
            )
            deltas.append(delta)
            indexes.append(torch.Tensor(shift_indexes))
        x, delta, indexes = self._pad_samples_same_length(X, deltas, indexes)
        return x, delta, indexes
