"""
Marek Krcál, Ondrej Svec, Martin Bálek, and Otakar Jasek.
Deep convolutional malware classifiers can learn from raw executables and labels only.
ICLR 2018. Workshop Track Proceedings, 2018.
https://openreview.net/pdf?id=HkHrmM1PM
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from maltorch.zoo.model import EmbeddingModel


def vec_bin_array(arr, m=8):
    """
    Arguments:
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[..., bit_ix] = fetch_bit_func(strs).astype(np.int8)

    return (ret * 2 - 1).astype(np.float32) / 16


class AvastStyleConv(EmbeddingModel):
    def __init__(self, embedding_size: int = 8, max_len: int = 512000, threshold: float = 0.5, padding_idx: int = 256,
                 is_embedding_fixed: bool = True, channels: int = 128, window_size: int = 32, stride: int = 4):
        super(AvastStyleConv, self).__init__(
            name="AvastStyleConv", gdrive_id=None
        )
        self.max_len = max_len
        self.threshold = threshold
        self.is_embedding_fixed = is_embedding_fixed
        self.invalid_value = padding_idx
        self._expansion = torch.tensor([[-1.0, 1.0]])
        self.channels = channels
        self.window_size = window_size
        self.stride = stride

        self.embedding_1 = nn.Embedding(
            num_embeddings=257, embedding_dim=embedding_size, padding_idx=padding_idx
        )
        if is_embedding_fixed:
            if padding_idx == 256:
                for i in range(0, 256):
                    self.embedding_1.weight.data[i, :] = torch.tensor(vec_bin_array(np.asarray([i])))
            elif padding_idx == 0:
                for i in range(1, 257):
                    self.embedding_1.weight.data[i, :] = torch.tensor(vec_bin_array(np.asarray([i])))
            else:
                raise NotImplementedError
            for param in self.embedding_1.parameters():
                param.requires_grad = False

        self.conv1d_1 = nn.Conv1d(8, self.channels, self.window_size, stride=self.stride, bias=True)
        self.conv1d_2 = nn.Conv1d(self.channels, self.channels * 2, self.window_size, stride=self.stride, bias=True)
        self.pool_1 = nn.MaxPool1d(4)
        self.conv1d_3 = nn.Conv1d(self.channels * 2, self.channels * 3, self.window_size // 2, stride=self.stride * 2,
                                  bias=True)
        self.conv1d_4 = nn.Conv1d(self.channels * 3, self.channels * 4, self.window_size // 2, stride=self.stride * 2,
                                  bias=True)

        self.dense_1 = nn.Linear(self.channels * 4, self.channels * 4)
        self.dense_2 = nn.Linear(self.channels * 4, self.channels * 3)
        self.dense_3 = nn.Linear(self.channels * 3, self.channels * 2)
        self.dense_4 = nn.Linear(self.channels * 2, 1)

    def embedding_layer(self):
        return self.embedding_1

    def embed(self, x):
        if self.is_embedding_fixed:
            with torch.no_grad():
                emb_x = self.embedding_1(x)
                emb_x = emb_x.transpose(1, 2)
        else:
            emb_x = self.embedding_1(x)
            emb_x = emb_x.transpose(1, 2)
        return emb_x

    def embedding_matrix(self):
        return self.embedding_1.weight

    def _forward_embed_x(self, x):
        conv1d_1 = torch.relu(self.conv1d_1(x))
        conv1d_2 = torch.relu(self.conv1d_2(conv1d_1))
        pool_1 = self.pool_1(conv1d_2)
        conv1d_3 = torch.relu(self.conv1d_3(pool_1))
        conv1d_4 = torch.relu(self.conv1d_4(conv1d_3))

        global_max_pooling1d_1 = F.max_pool1d(
            input=conv1d_4, kernel_size=conv1d_4.size()[2:]
        )
        global_max_pooling1d_1_flatten = global_max_pooling1d_1.view(
            global_max_pooling1d_1.size(0), -1
        )

        dense_1 = self.dense_1(global_max_pooling1d_1_flatten)
        dense_1_activation = torch.selu(dense_1)
        dense_2 = self.dense_2(dense_1_activation)
        dense_2_activation = torch.selu(dense_2)
        dense_3 = self.dense_3(dense_2_activation)
        dense_3_activation = torch.selu(dense_3)
        dense_4 = self.dense_4(dense_3_activation)
        return dense_4
