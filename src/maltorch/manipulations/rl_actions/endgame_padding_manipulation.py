import torch
import random
from maltorch.manipulations.byte_manipulation import ByteManipulation

class EndgamePaddingManipulation(ByteManipulation):
    """Append a random byte sequence to the input.

    The length of the padding is chosen uniformly in ``[2**5, 2**8]``.
    """

    def initialize(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # No delta needed for padding – return empty tensor
        zero_delta = torch.zeros((0,), dtype=torch.uint8, device=samples.device)
        return samples, zero_delta

    def _apply_manipulation(self, x: torch.Tensor, delta: torch.Tensor):
        samples = x.unsqueeze(0) if x.ndim == 1 else x
        batch_size, _ = samples.shape
        pad_len = 2 ** random.randint(5, 8)
        pad = torch.randint(0, 256, (batch_size, pad_len), dtype=torch.uint8, device=x.device)
        x_new = torch.cat([x.unsqueeze(0), pad], dim=1)
        return x_new, delta
