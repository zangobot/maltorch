from abc import ABC, abstractmethod

import torch
from secmlt.optimization.initializer import Initializer


class ByteBasedInitializer(Initializer, ABC):
    def __init__(self, random_init: bool = False):
        self.random_init = random_init

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor, list]: ...
