from typing import Callable, Optional

import torch.optim
from torch import Tensor

from maltorch.optim.base import BaseByteOptimizer

INVALID = torch.inf


class BGD(BaseByteOptimizer):
    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        for group in self.param_groups:
            for delta in group["params"]:
                if self._embedding_grad is None:
                    raise ValueError(
                        "Grad is none, you should call backward first before invoking the step function. "
                        "Otherwise, the model itself is not differentiable and gradients can not be computed."
                    )
                grad_embedding = self._embedding_grad[0]
                # grad_embedding = self.gradient_processing(grad_embedding)
                updated_delta = token_gradient_descent(
                    embedding_tokens=self.embedding_matrix,
                    emb_x=self.embedding_matrix[delta.long()],
                    gradient_f=grad_embedding,
                    step_size=self.step_size,
                    index_to_perturb=self.indexes_to_perturb,
                    unavailable_tokens=torch.LongTensor([256]),
                )
                updated_delta = updated_delta.to(delta.device)
                to_update = updated_delta != INVALID
                if torch.any(to_update):
                    delta.data[to_update] = updated_delta[to_update]
        return None


def token_gradient_descent(
    embedding_tokens: torch.Tensor,
    emb_x: torch.Tensor,
    gradient_f: torch.Tensor,
    step_size: int,
    index_to_perturb: torch.LongTensor,
    unavailable_tokens: Optional[torch.Tensor] = None,
) -> Tensor:
    optimized_tokens = torch.zeros(*emb_x.shape[:2]) + INVALID
    for j in range(emb_x.shape[0]):
        step_size_index = (
            gradient_f[j, :][index_to_perturb[j].view(-1)]
            .norm(dim=1)
            .argsort(descending=True)[:step_size]
        )
        for i in step_size_index:
            byte_in_grad = index_to_perturb[j, i]
            gradient_f_i = -gradient_f[j, byte_in_grad] #maltorch is written to minimize the loss
            x_i = emb_x[j, i]
            token_to_chose = single_token_gradient_update(
                start_token=x_i,
                gradient=gradient_f_i,
                embedded_tokens=embedding_tokens,
                unavailable_tokens=unavailable_tokens,
            )
            optimized_tokens[j, i] = token_to_chose
    return optimized_tokens


def single_token_gradient_update(
    start_token: torch.Tensor,
    gradient: torch.Tensor,
    embedded_tokens: torch.Tensor,
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
    if torch.equal(gradient, torch.zeros_like(gradient)):
        return invalid_val
    gradient = gradient / gradient.norm()
    B = embedded_tokens
    distance = torch.full((B.shape[0],), invalid_val, device=B.device)
    same_mask = torch.all(B == start_token, dim=1)
    if unavailable_tokens is not None:
        unavail_mask = torch.zeros_like(distance, dtype=torch.bool)
        unavail_mask[unavailable_tokens] = True
    else:
        unavail_mask = torch.zeros_like(distance, dtype=torch.bool)

    W = B - start_token  # (N, D)
    s = torch.sum(W * gradient, dim=1)  # (N,)
    forward_mask = s >= 0
    valid_mask = (~same_mask) & (~unavail_mask) & forward_mask
    if not torch.any(valid_mask):
        return invalid_val

    proj = start_token + s[valid_mask].unsqueeze(1) * gradient  # (Nv, D)
    dist_valid = torch.norm(B[valid_mask] - proj, dim=1)
    distance[valid_mask] = dist_valid
    min_value, token_to_choose = torch.min(distance, dim=0)

    if min_value == invalid_val:
        return invalid_val

    return token_to_choose
