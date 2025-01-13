from abc import ABC
from typing import Union, Iterable, Dict, Any

from secmlt.optimization.gradient_processing import GradientProcessing
from torch import Tensor
import torch.optim

from maltorch.zoo.model import BaseEmbeddingPytorchClassifier


class BaseByteOptimizer(torch.optim.Optimizer, ABC):
    def __init__(
        self,
        params: Union[Iterable[Tensor], Iterable[Dict[str, Any]]],
        model: BaseEmbeddingPytorchClassifier,
        indexes_to_perturb: torch.LongTensor,
        gradient_processing: GradientProcessing,
        lr: int = 16,
        device: str = "cpu",
    ):
        defaults = {
            "byte_step_size": lr,
            "embedding_matrix": model.embedding_matrix(),
        }
        super().__init__(params, defaults)
        self.gradient_processing = gradient_processing
        self.step_size = lr
        self.embedding_matrix = model.embedding_matrix()
        self.indexes_to_perturb = indexes_to_perturb
        self.device = device
        model.embedding_layer().register_full_backward_hook(self._backward_hook)
        self._embedding_grad = None

    def zero_grad(self, set_to_none: bool = ...) -> None:
        super().zero_grad(set_to_none=set_to_none)
        self._embedding_grad = None

    def _backward_hook(self, module, input_grad, output_grad):
        self._embedding_grad = output_grad
