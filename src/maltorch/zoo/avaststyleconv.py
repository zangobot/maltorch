"""
Marek Krcál, Ondrej Svec, Martin Bálek, and Otakar Jasek.
Deep convolutional malware classifiers can learn from raw executables and labels only.
ICLR 2018. Workshop Track Proceedings, 2018.
https://openreview.net/pdf?id=HkHrmM1PM
"""

import torch
import torch.nn.functional as F
from torch import nn
from maltorch.zoo.model import EmbeddingModel


class AvastStyleConv(EmbeddingModel):
    def __init__(self,
                 embedding_size: int = 8,
                 max_len: int = 512000,
                 threshold: float = 0.5,
                 padding_idx: int = 256,
                 channels: int = 48,
                 window_size: int = 32,
                 stride: int = 4):
        super(AvastStyleConv, self).__init__(
            name="AvastStyleConv", gdrive_id=None
        )
        # "14wSZQ-Drns9G8CEqvfbAZjmMVt_ToUbp"
        self.max_len = max_len
        self.threshold = threshold
        self.invalid_value = padding_idx
        self.channels = channels
        self.window_size = window_size
        self.stride = stride

        self.embedding_1 = nn.Embedding(
            num_embeddings=257, embedding_dim=embedding_size, padding_idx=padding_idx
        )

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

        global_max_pooling1d_1 = F.avg_pool1d(
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
