import torch
from secml2.optimization.initializer import Initializer


class ByteBasedInitializer(Initializer):
    def __init__(self, manipulation_size):
        self.manipulation_size = manipulation_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        init = torch.zeros((x.shape[0], self.manipulation_size), dtype=torch.float)
        return init
