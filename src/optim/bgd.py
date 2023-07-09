from typing import Union, Iterable, Dict, Any, Callable, Optional

import torch.optim
from torch import Tensor

from src.zoo.model import BaseEmbeddingPytorchClassifier

INVALID = torch.inf


class BGD(torch.optim.Optimizer):
    def __init__(
            self,
            params: Union[Iterable[Tensor], Iterable[Dict[str, Any]]],
            model: BaseEmbeddingPytorchClassifier,
            indexes_to_perturb: torch.LongTensor,
            lr: int = 16,
            device: str = "cpu",
    ):
        defaults = {
            "byte_step_size": lr,
            "embedding_matrix": model.embedding_matrix(),
        }
        super().__init__(params, defaults)
        self.step_size = lr
        self.embedding_matrix = model.embedding_matrix()
        self.device = device
        self.indexes_to_perturb = indexes_to_perturb
        model.embedding_layer().register_full_backward_hook(self._backward_hook)
        self._embedding_grad = None

    def zero_grad(self, set_to_none: bool = ...) -> None:
        super().zero_grad(set_to_none=set_to_none)
        self._embedding_grad = None

    def _backward_hook(self, module, input_grad, output_grad):
        self._embedding_grad = output_grad

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        for group in self.param_groups:
            for delta in group["params"]:
                grad_embedding = self._embedding_grad[0]
                if self._embedding_grad is None:
                    raise ValueError("Grad is none, you should call backward first before invoking the step function.")
                updated_delta = token_gradient_descent(
                    embedding_tokens=self.embedding_matrix,
                    emb_x=self.embedding_matrix[delta.long()],
                    gradient_f=grad_embedding,
                    index_to_perturb=self.indexes_to_perturb,
                    step_size=self.step_size,
                    admitted_tokens=torch.LongTensor(range(0, 256)),
                    unavailable_tokens=torch.LongTensor([256]),
                )
                to_update = updated_delta != INVALID
                delta.data[self.indexes_to_perturb[to_update], :] = updated_delta[to_update]
        return None


def token_gradient_descent(
        embedding_tokens: torch.Tensor,
        emb_x: torch.Tensor,
        gradient_f: torch.Tensor,
        index_to_perturb: torch.LongTensor,
        step_size: int,
        admitted_tokens: torch.LongTensor,
        unavailable_tokens: Optional[torch.Tensor] = None,
) -> Tensor:
    optimized_tokens = torch.zeros(*emb_x.shape[:2])
    step_size_index = gradient_f[0, index_to_perturb].norm(dim=1).argsort()[:step_size]
    step_size_index = torch.LongTensor(index_to_perturb)[step_size_index]
    for i in step_size_index:
        gradient_f_i = gradient_f[0, i]
        x_i = emb_x[0, i]
        token_to_chose = single_token_gradient_update(
            start_token=x_i.cpu(),
            gradient=gradient_f_i.cpu(),
            embedded_tokens=embedding_tokens.cpu(),
            admitted_tokens=admitted_tokens.cpu(),
            unavailable_tokens=unavailable_tokens,
        )
        optimized_tokens[i] = token_to_chose
    return optimized_tokens


def single_token_gradient_update(
        start_token: torch.Tensor,
        gradient: torch.Tensor,
        embedded_tokens: torch.Tensor,
        admitted_tokens: torch.LongTensor,
        invalid_val=INVALID,
        unavailable_tokens: Optional[torch.Tensor] = None,
):
    """
    Given the starting byte, the gradient and the embedding map,it returns a list of distances

    Parameters
    ----------
    start_token : torch.Tensor
        the starting embedding token for the search
    gradient : torch.Tensor
        the gradient of a single embedded token
    embedded_tokens : torch.Tensor
        the embedding matrix with all the byte embedded
    admitted_tokens: list
        the list of indexes of the tokens to use in the search
    invalid_val : optional, default torch.inf
        the invalid value to use. Default torch.inf
    unavailable_tokens: Union[torch.Tensor, None] = None
        if specified, it avoids the usage of the selected tokens during the search step
    Returns
    -------

    """
    if torch.equal(gradient, torch.zeros(gradient.shape)):
        return INVALID
    distance = torch.zeros(len(admitted_tokens))
    # gs = gradient / torch.norm(gradient)
    for i, b in enumerate(embedded_tokens[admitted_tokens, :].cpu()):
        if torch.all(start_token == b):
            distance[i] = invalid_val
            continue
        if unavailable_tokens is not None:
            if i in unavailable_tokens:
                distance[i] = INVALID
        bts = b - start_token
        s_i = torch.dot(gradient, bts)
        if s_i <= 0:
            distance[i] = invalid_val
        else:
            d_i = torch.norm(b - (start_token + s_i * gradient))
            distance[i] = d_i
    min_value, token_to_chose = torch.min(distance, dim=0, keepdim=True)
    if min_value == INVALID:
        return INVALID
    return token_to_chose
