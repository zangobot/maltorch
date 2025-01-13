from abc import ABC, abstractmethod

import torch
from secmlt.optimization.initializer import Initializer


class ByteBasedInitializer(Initializer, ABC):
    def __init__(self, random_init: bool = False):
        self.random_init = random_init

    @staticmethod
    def _pad_samples_same_length(list_of_samples, list_of_deltas, list_of_indexes):
        x = (
            torch.nn.utils.rnn.pad_sequence(list_of_samples, padding_value=256)
            .transpose(0, 1)
            .float()
        )
        delta = (
            torch.nn.utils.rnn.pad_sequence(list_of_deltas, padding_value=256)
            .transpose(0, 1)
            .float()
        )
        indexes = (
            torch.nn.utils.rnn.pad_sequence(list_of_indexes, padding_value=-1)
            .transpose(0, 1)
            .long()
        )
        return x, delta, indexes

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor, list]: ...


class IdentityInitializer(ByteBasedInitializer):
    def __init__(self, random_init: bool = False):
        super().__init__(random_init=random_init)

    def __call__(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor, list]:
        delta = torch.rand_like(x) if self.random_init else torch.zeros_like(x)
        return x, delta, []
