import torch

from maltorch.initializers.initializers import ByteBasedInitializer
from maltorch.utils.pe_operations import padding_manipulation


class PaddingInitializer(ByteBasedInitializer):
    def __init__(self, padding: int = 2048, random_init: bool = False):
        super().__init__(random_init)
        self.padding = padding

    def __call__(self, x: torch.Tensor):
        X = []
        deltas = []
        indexes = []
        for x_i in x:
            x_i, padding_indexes = padding_manipulation(x_i, self.padding)
            X.append(x_i)
            delta = (
                torch.zeros(len(padding_indexes))
                if not self.random_init
                else torch.randint(0, 255, (len(padding_indexes),))
            )
            deltas.append(delta)
            indexes.append(padding_indexes)
        x, delta, indexes = self._pad_samples_same_length(X, deltas, indexes)
        return x, delta, indexes
