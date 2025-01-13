import torch

from maltorch.initializers.initializers import ByteBasedInitializer


class PartialDOSInitializer(ByteBasedInitializer):
    def __init__(self, random_init: bool = False):
        super().__init__(random_init)

    def __call__(self, x: torch.Tensor):
        indexes = torch.arange(2, 60).long().repeat((x.shape[0], 1))
        delta = (
            torch.zeros((x.shape[0], 58))
            if not self.random_init
            else torch.randint(0, 255, (x.shape[0], 58))
        ).float()
        return x, delta, indexes
