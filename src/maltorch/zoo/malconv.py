"""
Malware Detection by Eating a Whole EXE
Edward Raff, Jon Barker, Jared Sylvester, Robert Brandon, Bryan Catanzaro, Charles Nicholas
https://arxiv.org/abs/1710.09435
"""

import torch
import torch.nn.functional as F
from torch import nn

from maltorch.zoo.model import EmbeddingModel


class MalConv(EmbeddingModel):
    """
    Architecture implementation.
    """

    def __init__(self, embedding_size: int = 8,
                 max_len: int =2**20,
                 threshold: float =0.5,
                 padding_idx: int = 256):
        super(MalConv, self).__init__(
            name="MalConv", gdrive_id="1Hg8I7Jx13LmnSPBjsPGr8bvmmS874Y9N"
        )
        self.embedding_1 = nn.Embedding(
            num_embeddings=257, embedding_dim=embedding_size, padding_idx=padding_idx
        )
        self.conv1d_1 = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=128,
            kernel_size=(500,),
            stride=(500,),
            groups=1,
            bias=True,
        )
        self.conv1d_2 = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=128,
            kernel_size=(500,),
            stride=(500,),
            groups=1,
            bias=True,
        )
        self.dense_1 = nn.Linear(in_features=128, out_features=128, bias=True)
        self.dense_2 = nn.Linear(in_features=128, out_features=1, bias=True)
        self.embedding_size = (embedding_size,)
        self.max_len = max_len
        self.threshold = threshold
        self.invalid_value = padding_idx
        self._expansion = torch.tensor([[-1.0, 1.0]])

    def embedding_layer(self):
        return self.embedding_1

    def embed(self, x):
        emb_x = self.embedding_1(x)
        emb_x = emb_x.transpose(1, 2)
        return emb_x

    def _forward_embed_x(self, x):
        conv1d_1 = self.conv1d_1(x)
        conv1d_2 = self.conv1d_2(x)
        conv1d_1_activation = torch.relu(conv1d_1)
        conv1d_2_activation = torch.sigmoid(conv1d_2)
        multiply_1 = conv1d_1_activation * conv1d_2_activation
        global_max_pooling1d_1 = F.max_pool1d(
            input=multiply_1, kernel_size=multiply_1.size()[2:]
        )
        global_max_pooling1d_1_flatten = global_max_pooling1d_1.view(
            global_max_pooling1d_1.size(0), -1
        )
        dense_1 = self.dense_1(global_max_pooling1d_1_flatten)
        dense_1_activation = torch.relu(dense_1)
        dense_2 = self.dense_2(dense_1_activation)
        return dense_2

    def embedding_matrix(self):
        return self.embedding_1.weight
