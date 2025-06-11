"""
Coull, Scott E., and Christopher Gardner.
"Activation analysis of a byte-based deep neural network for malware classification."
2019 IEEE Security and Privacy Workshops (SPW). IEEE, 2019.
"""


import torch

from maltorch.zoo.model import EmbeddingModel


class BBDnn(EmbeddingModel):
    def __init__(
            self,
            embedding_size: int = 10,
            min_len: int = 4096,
            max_len: int = 102400,
            threshold: float = 0.5,
            padding_idx: int = 256,
    ):
        super(BBDnn, self).__init__(name="bbdnn", gdrive_id="1c_9lVHT9zYpBCwQfnUW6ZbCF6SaVabRZ", min_len=min_len, max_len=max_len)
        self.max_len = max_len
        self.threshold = threshold
        self.embedding_1 = torch.nn.Embedding(
            num_embeddings=257, embedding_dim=embedding_size, padding_idx=padding_idx
        )
        self.conv1d_1 = torch.nn.Conv1d(
            in_channels=embedding_size,
            out_channels=96,
            kernel_size=(11,),
            stride=(1,),
            groups=1,
            bias=True,
        )
        self.conv1d_2 = torch.nn.Conv1d(
            in_channels=96,
            out_channels=128,
            kernel_size=(5,),
            stride=(1,),
            groups=1,
            bias=True,
        )
        self.conv1d_3 = torch.nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=(5,),
            stride=(1,),
            groups=1,
            bias=True,
        )
        self.conv1d_4 = torch.nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=(5,),
            stride=(1,),
            groups=1,
            bias=True,
        )
        self.conv1d_5 = torch.nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=(5,),
            stride=(1,),
            groups=1,
            bias=True,
        )
        self.dense_1 = torch.nn.Linear(in_features=512, out_features=1, bias=True)

    def embedding_layer(self):
        return self.embedding_1

    def _forward_embed_x(self, x):
        conv1d_1 = self.conv1d_1(x)
        conv1d_1_activation = torch.relu(conv1d_1)
        max_pooling1d_1 = torch.max_pool1d(
            conv1d_1_activation,
            kernel_size=(4,),
            stride=(4,),
            padding=0,
            ceil_mode=False,
        )

        conv1d_2 = self.conv1d_2(max_pooling1d_1)
        conv1d_2_activation = torch.relu(conv1d_2)
        max_pooling1d_2 = torch.max_pool1d(
            conv1d_2_activation,
            kernel_size=(4,),
            stride=(4,),
            padding=0,
            ceil_mode=False,
        )

        conv1d_3 = self.conv1d_3(max_pooling1d_2)
        conv1d_3_activation = torch.relu(conv1d_3)
        max_pooling1d_3 = torch.max_pool1d(
            conv1d_3_activation,
            kernel_size=(4,),
            stride=(4,),
            padding=0,
            ceil_mode=False,
        )

        conv1d_4 = self.conv1d_4(max_pooling1d_3)
        conv1d_4_activation = torch.relu(conv1d_4)
        max_pooling1d_4 = torch.max_pool1d(
            conv1d_4_activation,
            kernel_size=(4,),
            stride=(4,),
            padding=0,
            ceil_mode=False,
        )

        conv1d_5 = self.conv1d_5(max_pooling1d_4)
        conv1d_5_activation = torch.relu(conv1d_5)
        global_max_pooling1d_1 = torch.max_pool1d(
            input=conv1d_5_activation, kernel_size=conv1d_5_activation.size()[2:]
        )
        global_average_pooling1d_1 = torch.avg_pool1d(
            input=conv1d_5_activation, kernel_size=conv1d_5_activation.size()[2:]
        )
        global_max_pooling1d_1_flatten = global_max_pooling1d_1.view(
            global_max_pooling1d_1.size(0), -1
        )
        global_average_pooling1d_1_flatten = global_average_pooling1d_1.view(
            global_average_pooling1d_1.size(0), -1
        )
        concatenate_1 = torch.cat(
            (global_max_pooling1d_1_flatten, global_average_pooling1d_1_flatten), 1
        )
        dense_1 = self.dense_1(concatenate_1)
        return dense_1

    def embed(self, x):
        emb_x = self.embedding_1(x)
        emb_x = emb_x.transpose(1, 2)
        return emb_x

    def embedding_matrix(self):
        return self.embedding_1.weight
