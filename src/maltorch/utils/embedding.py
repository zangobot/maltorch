import copy
from typing import Optional

import torch
from torch import Tensor


class DifferentiableOneHotEncoding(torch.nn.Module):
    def __init__(self, classes):
        super(DifferentiableOneHotEncoding, self).__init__()
        self.num_classes = classes

    def forward(self, x):
        eye = torch.eye(self.num_classes, device=x.device)
        one_hot = torch.index_select(eye, dim=0, index=x.view(-1))
        one_hot = one_hot.view(x.shape[0], x.shape[1], self.num_classes)
        return one_hot


class IdentityEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            _freeze,
            device,
            dtype,
        )

        self.encoder = DifferentiableOneHotEncoding(classes=self.num_embeddings)

    def forward(self, x: Tensor) -> Tensor:
        x_encoded = self.encoder(x)
        emb = x_encoded.matmul(self.weight)
        return emb


def patch_embedding_backward(model: torch.nn.Module):
    patched_model = copy.deepcopy(model)
    modules = patched_model.named_modules()
    for name, module in modules:
        if isinstance(module, torch.nn.Embedding):
            new_embedding_layer = IdentityEmbedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                padding_idx=module.padding_idx,
                max_norm=module.max_norm,
                norm_type=module.norm_type,
                scale_grad_by_freq=module.scale_grad_by_freq,
                sparse=module.sparse,
                device=module.weight.device,
                _weight=module.weight,
            )
            setattr(patched_model, name, new_embedding_layer)
    return patched_model
