"""
Daniel Gibert, Carles Mateu, Jordi Planes, Daniel Solis, Ramon Vicens
"Convolutional neural networks for classification of malware assembly code"
Recent Advances in Artificial Intelligence Research and Development: Proceedings of the 20th International Conference
of the Catalan Association for Artificial Intelligence.

Niall McLaughlin, Jesús Martínez del Rincón, BooJoong Kang, Suleiman Y. Yerima, Paul Miller, Sakir Sezer,
Yeganeh Safaei, Erik Trickel, Ziming Zhao, Adam Doupé, Gail-Joon Ahn
Deep Android Malware Detection.
ACM Conference on Data and Application Security and Privacy (CODASPY 2017)

Loosely based on the shallow convolutional neural network architecture defined in Gibert et al. 2017 and
McLaughlin et al. 2017 but with only a convolutional layer with kernel size equals to 3.
"""

import torch
import torch.nn.functional as F
from torch import nn
from maltorch.zoo.model import EmbeddingModel


class NGramConv(EmbeddingModel):
    def __init__(self,
                 embedding_size=8,
                 max_len=2 ** 20,
                 out_channels: int = 100,
                 threshold: float = 0.5,
                 padding_idx: int = 256):
        super(NGramConv, self).__init__(
            name="NGramConv", gdrive_id=None
        )
        self.embedding_1 = nn.Embedding(
            num_embeddings=257, embedding_dim=embedding_size, padding_idx=padding_idx
        )
        self.out_channels = out_channels
        self.conv1d_1 = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=self.out_channels,
            kernel_size=(7,),
            stride=(3,),
            groups=1,
            bias=True,
        )

        self.dense_1 = nn.Linear(in_features=self.out_channels, out_features=self.out_channels, bias=True)
        self.drop_1 = nn.Dropout1d()
        self.dense_2 = nn.Linear(in_features=self.out_channels, out_features=1, bias=True)

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
        conv1d_1_activation = torch.relu(conv1d_1)

        global_max_pooling1d_1 = F.max_pool1d(
            input=conv1d_1_activation, kernel_size=conv1d_1_activation.size()[2:]
        )
        global_max_pooling1d_1_flatten = global_max_pooling1d_1.view(
            global_max_pooling1d_1.size(0), -1
        )

        dense_1 = self.dense_1(global_max_pooling1d_1_flatten)
        dense_1_activation = torch.relu(dense_1)
        drop_1 = self.drop_1(dense_1_activation)
        dense_2 = self.dense_2(drop_1)
        return dense_2

    def embedding_matrix(self):
        return self.embedding_1.weight
