from abc import ABC, abstractmethod

import torch
from secml2.optimization.initializer import Initializer

from src.utils.pe_operations import content_shift_manipulation


class ByteBasedInitializer(Initializer, ABC):
    def __init__(self, random_init: bool = False):
        self.random_init = random_init

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor, list]:
        ...


class PartialDOSInitializer(ByteBasedInitializer):
    def __init__(self, random_init: bool = False):
        super().__init__(random_init)

    def __call__(self, x: torch.Tensor):
        indexes = torch.arange(2, 60).long().repeat((x.shape[0], 1))
        delta = (
            torch.zeros((x.shape[0], 58))
            if not self.random_init
            else torch.randint(0, 255, (x.shape[0], 58))
        )
        return x, delta, indexes


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
