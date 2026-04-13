from typing import Callable, Optional

import torch
from torch import Tensor

from maltorch.optim.base import BaseByteOptimizer

INVALID = torch.inf


class BGD(BaseByteOptimizer):
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if self._embedding_grad is None:
            raise ValueError(
                "Grad is none, you should call backward first before invoking the step function. "
                "Otherwise, the model itself is not differentiable and gradients can not be computed."
            )

        grad_embedding = self._embedding_grad[0]

        for group in self.param_groups:
            for delta in group["params"]:
                device = delta.device

                token_ids = delta.long().to(device)
                emb_x = self.embedding_matrix[token_ids].to(device)

                updated_delta = token_gradient_descent(
                    embedding_tokens=self.embedding_matrix.to(device),
                    token_ids=token_ids,
                    emb_x=emb_x,
                    gradient_f=grad_embedding.to(device),
                    step_size=self.step_size,
                    index_to_perturb=self.indexes_to_perturb.to(device=device, dtype=torch.long),
                    unavailable_tokens=torch.tensor([256], device=device, dtype=torch.long),
                )

                to_update = updated_delta != INVALID
                if torch.any(to_update):
                    delta.data[to_update] = updated_delta[to_update].to(delta.dtype)

        return None


@torch.no_grad()
def token_gradient_descent(
    embedding_tokens: torch.Tensor,
    token_ids: torch.Tensor,
    emb_x: torch.Tensor,
    gradient_f: torch.Tensor,
    step_size: int,
    index_to_perturb: torch.LongTensor,
    unavailable_tokens: Optional[torch.Tensor] = None,
) -> Tensor:
    """
    Parameters
    ----------
    embedding_tokens : [V, D]
        Embedding matrix
    token_ids : [B, P]
        Current delta token ids
    emb_x : [B, P, D]
        Embedded delta tokens
    gradient_f : [B, L, D]
        Gradient wrt the full sequence embeddings
    step_size : int
        Number of perturbable positions to update per sample
    index_to_perturb : [B, P]
        Absolute positions in the full sequence, padded with -1
    unavailable_tokens : [K] or None
        Token ids that cannot be selected

    Returns
    -------
    optimized_tokens : [B, P]
        Selected token ids at updated relative positions, INVALID elsewhere
    """
    device = gradient_f.device

    embedding_tokens = embedding_tokens.to(device)
    token_ids = token_ids.to(device=device, dtype=torch.long)
    emb_x = emb_x.to(device)
    index_to_perturb = index_to_perturb.to(device=device, dtype=torch.long)
    if unavailable_tokens is not None:
        unavailable_tokens = unavailable_tokens.to(device=device, dtype=torch.long)

    batch_size, num_delta_pos, emb_dim = emb_x.shape
    grad_len = gradient_f.shape[1]

    optimized_tokens = torch.full(
        (batch_size, num_delta_pos),
        INVALID,
        device=device,
        dtype=torch.float32,
    )

    if step_size <= 0 or num_delta_pos == 0:
        return optimized_tokens

    # valid positions are those that are both non-negative and inside gradient_f
    valid_pos_mask = (index_to_perturb >= 0) & (index_to_perturb < grad_len)  # [B, P]

    if not torch.any(valid_pos_mask):
        return optimized_tokens

    # IMPORTANT:
    # invalid indices must remain logically invalid.
    # we replace them with 0 only as a technical placeholder for gather.
    safe_index_to_perturb = torch.where(
        valid_pos_mask,
        index_to_perturb,
        torch.zeros_like(index_to_perturb, device=device),
    )  # [B, P]

    # gather gradients from full sequence using absolute positions
    gather_idx = safe_index_to_perturb.unsqueeze(-1).expand(-1, -1, emb_dim)  # [B, P, D]
    grad_on_perturb = gradient_f.gather(1, gather_idx)  # [B, P, D]

    # invalidate fake positions so they are never chosen by topk
    grad_norms = grad_on_perturb.norm(dim=2)  # [B, P]
    grad_norms = grad_norms.masked_fill(~valid_pos_mask, -1.0)

    k = min(step_size, num_delta_pos)

    # select perturbation slots in delta-space
    selected_rel_idx = grad_norms.topk(k, dim=1, largest=True, sorted=False).indices  # [B, k]

    # mask of selected entries that are truly valid
    selected_valid_mask = valid_pos_mask.gather(1, selected_rel_idx)  # [B, k]

    # absolute positions are used ONLY for gradient_f
    selected_abs_pos = safe_index_to_perturb.gather(1, selected_rel_idx)  # [B, k]

    grad_selected = -gradient_f.gather(
        1, selected_abs_pos.unsqueeze(-1).expand(-1, -1, emb_dim)
    )  # [B, k, D]

    # local delta-space positions are used for emb_x and token_ids
    start_tokens = emb_x.gather(
        1, selected_rel_idx.unsqueeze(-1).expand(-1, -1, emb_dim)
    )  # [B, k, D]

    start_token_ids = token_ids.gather(1, selected_rel_idx)  # [B, k]

    chosen = batched_single_token_gradient_update(
        start_tokens=start_tokens.reshape(-1, emb_dim),
        start_token_ids=start_token_ids.reshape(-1),
        gradients=grad_selected.reshape(-1, emb_dim),
        embedded_tokens=embedding_tokens,
        unavailable_tokens=unavailable_tokens,
    ).view(batch_size, k)

    # invalidate fake selections
    chosen = chosen.masked_fill(~selected_valid_mask, INVALID)

    # write back in delta-space
    optimized_tokens.scatter_(1, selected_rel_idx, chosen)

    return optimized_tokens


@torch.no_grad()
def batched_single_token_gradient_update(
    start_tokens: torch.Tensor,       # [K, D]
    start_token_ids: torch.Tensor,    # [K]
    gradients: torch.Tensor,          # [K, D]
    embedded_tokens: torch.Tensor,    # [V, D]
    unavailable_tokens: Optional[torch.Tensor] = None,
    invalid_val: float = INVALID,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Batched version of single_token_gradient_update.

    Returns
    -------
    out : [K]
        chosen token ids as float tensor, or INVALID if no valid token exists
    """
    device = embedded_tokens.device

    start_tokens = start_tokens.to(device)
    start_token_ids = start_token_ids.to(device=device, dtype=torch.long)
    gradients = gradients.to(device)
    if unavailable_tokens is not None:
        unavailable_tokens = unavailable_tokens.to(device=device, dtype=torch.long)

    vocab_size, emb_dim = embedded_tokens.shape

    out = torch.full(
        (start_tokens.shape[0],),
        invalid_val,
        device=device,
        dtype=torch.float32,
    )

    grad_norms = gradients.norm(dim=1, keepdim=True)
    nonzero_mask = grad_norms.squeeze(1) > 0

    if not torch.any(nonzero_mask):
        return out

    starts = start_tokens[nonzero_mask]                        # [Q, D]
    grads = gradients[nonzero_mask] / grad_norms[nonzero_mask]
    ids = start_token_ids[nonzero_mask]                       # [Q]

    B = embedded_tokens                                        # [V, D]
    B_norm2 = (B * B).sum(dim=1)                               # [V]
    token_range = torch.arange(vocab_size, device=device, dtype=torch.long)

    unavailable_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    if unavailable_tokens is not None and unavailable_tokens.numel() > 0:
        unavailable_mask[unavailable_tokens] = True

    chosen_nonzero = torch.full(
        (starts.shape[0],),
        invalid_val,
        device=device,
        dtype=torch.float32,
    )

    for begin in range(0, starts.shape[0], chunk_size):
        end = min(begin + chunk_size, starts.shape[0])

        s_chunk = starts[begin:end]       # [C, D]
        g_chunk = grads[begin:end]        # [C, D]
        id_chunk = ids[begin:end]         # [C]

        # s = <B - start, g> = <B, g> - <start, g>
        Bg = B @ g_chunk.T                                   # [V, C]
        start_dot_g = (s_chunk * g_chunk).sum(dim=1)         # [C]
        s = Bg.T - start_dot_g.unsqueeze(1)                  # [C, V]

        # ||B - start||^2
        Bs = B @ s_chunk.T                                   # [V, C]
        start_norm2 = (s_chunk * s_chunk).sum(dim=1)         # [C]
        w_norm2 = (
            B_norm2.unsqueeze(0)
            + start_norm2.unsqueeze(1)
            - 2 * Bs.T
        )                                                    # [C, V]

        # distance to projection
        dist2 = w_norm2 - s * s

        valid_mask = s >= 0
        valid_mask &= token_range.unsqueeze(0) != id_chunk.unsqueeze(1)

        if unavailable_mask.any():
            valid_mask &= ~unavailable_mask.unsqueeze(0)

        dist2 = dist2.masked_fill(~valid_mask, invalid_val)

        best_dist2, best_token = dist2.min(dim=1)
        has_valid = torch.isfinite(best_dist2)

        chosen_nonzero[begin:end] = torch.where(
            has_valid,
            best_token.to(torch.float32),
            torch.full_like(best_token, invalid_val, dtype=torch.float32),
        )

    out[nonzero_mask] = chosen_nonzero
    return out


@torch.no_grad()
def single_token_gradient_update(
    start_token: torch.Tensor,
    gradient: torch.Tensor,
    embedded_tokens: torch.Tensor,
    invalid_val=INVALID,
    unavailable_tokens: Optional[torch.Tensor] = None,
):
    """
    Kept for compatibility/debugging.
    Single-sample version of the same update logic.
    """
    device = embedded_tokens.device

    start_token = start_token.to(device)
    gradient = gradient.to(device)
    if unavailable_tokens is not None:
        unavailable_tokens = unavailable_tokens.to(device=device, dtype=torch.long)

    if torch.equal(gradient, torch.zeros_like(gradient)):
        return invalid_val

    gradient = gradient / gradient.norm()
    B = embedded_tokens

    distance = torch.full((B.shape[0],), invalid_val, device=device)

    same_mask = torch.all(B == start_token, dim=1)

    if unavailable_tokens is not None:
        unavail_mask = torch.zeros(B.shape[0], dtype=torch.bool, device=device)
        unavail_mask[unavailable_tokens] = True
    else:
        unavail_mask = torch.zeros(B.shape[0], dtype=torch.bool, device=device)

    W = B - start_token
    s = torch.sum(W * gradient, dim=1)
    forward_mask = s >= 0
    valid_mask = (~same_mask) & (~unavail_mask) & forward_mask

    if not torch.any(valid_mask):
        return invalid_val

    proj = start_token + s[valid_mask].unsqueeze(1) * gradient
    dist_valid = torch.norm(B[valid_mask] - proj, dim=1)
    distance[valid_mask] = dist_valid

    min_value, token_to_choose = torch.min(distance, dim=0)

    if min_value == invalid_val:
        return invalid_val

    return token_to_choose