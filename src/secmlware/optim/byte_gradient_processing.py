import torch
from secmlt.optimization.gradient_processing import GradientProcessing
from torch.nn.functional import normalize


class ByteGradientProcessing(GradientProcessing):
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        grad.data = normalize(grad.data, p=2, dim=1)
        return grad
