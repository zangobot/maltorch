import torch

from secmlware.initializers.initializers import ByteBasedInitializer
from secmlware.utils.pe_operations import content_shift_manipulation


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
            indexes.append(shift_indexes)
        x = (
            torch.nn.utils.rnn.pad_sequence(X, padding_value=256)
            .transpose(0, 1)
            .float()
        )
        delta = (
            torch.nn.utils.rnn.pad_sequence(deltas, padding_value=256)
            .transpose(0, 1)
            .float()
        )
        indexes = (
            torch.nn.utils.rnn.pad_sequence(indexes, padding_value=-1)
            .transpose(0, 1)
            .long()
        )
        return x, delta, indexes
