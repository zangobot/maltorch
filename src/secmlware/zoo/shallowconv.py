"""
Daniel Gibert, Carles Mateu, Jordi Planes, Daniel Solis, Ramon Vicens
"Convolutional neural networks for classification of malware assembly code"
Recent Advances in Artificial Intelligence Research and Development: Proceedings of the 20th International Conference
of the Catalan Association for Artificial Intelligence.

Niall McLaughlin, Jesús Martínez del Rincón, BooJoong Kang, Suleiman Y. Yerima, Paul Miller, Sakir Sezer, Yeganeh Safaei,
 Erik Trickel, Ziming Zhao, Adam Doupé, Gail-Joon Ahn
Deep Android Malware Detection.
ACM Conference on Data and Application Security and Privacy (CODASPY 2017)

This implementation is loosely based on the shallow convolutional neural network architecture defined in
Gibert et al. 2017 and McLaughlin et al.2017 but with three convolutional layers with kernel sizes 3, 5, and 7.

"""

from typing import Optional, Union
import torch
import torch.nn.functional as F
from torch import nn
from secmlt.models.data_processing.data_processing import DataProcessing
from secmlt.models.base_trainer import BaseTrainer
from secmlware.zoo.model import EmbeddingModel, BaseEmbeddingPytorchClassifier

class ShallowConv(EmbeddingModel):
    @classmethod
    def create_model(
            cls,
            model_path: Optional[str] = None,
            device: str = "cpu",
            preprocessing: DataProcessing = None,
            postprocessing: DataProcessing = None,
            trainer: BaseTrainer = None,
            threshold: Optional[Union[float, None]] = 0.5,
            **kwargs,
    ) -> BaseEmbeddingPytorchClassifier:
        net = cls(**kwargs)
        net.load_pretrained_model(device=device, model_path=model_path)
        net.eval()
        net = BaseEmbeddingPytorchClassifier(
            model=net,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            trainer=trainer,
            threshold=threshold,
        )
        return net

    def __init__(self, embedding_size=8, max_input_size=2**20, out_channels: int = 100, threshold: float = 0.5, padding_value: int = 256):
        super(ShallowConv, self).__init__(
            name="ShallowConv", gdrive_id="ModelWeightsNotUploadedYet"
        )
        self.embedding_1 = nn.Embedding(
            num_embeddings=257, embedding_dim=embedding_size, padding_idx=padding_value
        )
        self.out_channels = out_channels
        self.conv1d_1 = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=self.out_channels ,
            kernel_size=(3,),
            stride=(1,),
            groups=1,
            bias=True,
        )

        self.conv1d_2 = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=self.out_channels ,
            kernel_size=(5,),
            stride=(1,),
            groups=1,
            bias=True,
        )

        self.conv1d_3 = nn.Conv1d(
            in_channels=embedding_size,
            out_channels=self.out_channels ,
            kernel_size=(7,),
            stride=(1,),
            groups=1,
            bias=True,
        )

        self.dense_1 = nn.Linear(in_features=self.out_channels*3, out_features=self.out_channels, bias=True)
        self.drop_1 = nn.Dropout1d()
        self.dense_2 = nn.Linear(in_features=self.out_channels, out_features=1, bias=True)

        self.embedding_size = (embedding_size,)
        self.max_input_size = max_input_size
        self.threshold = threshold
        self.invalid_value = padding_value
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
        conv1d_3 = self.conv1d_3(x)
        conv1d_1_activation = torch.relu(conv1d_1)
        conv1d_2_activation = torch.relu(conv1d_2)
        conv1d_3_activation = torch.relu(conv1d_3)

        global_max_pooling1d_1 = F.max_pool1d(
            input=conv1d_1_activation, kernel_size=conv1d_1_activation.size()[2:]
        )
        global_max_pooling1d_2 = F.max_pool1d(
            input=conv1d_2_activation, kernel_size=conv1d_2_activation.size()[2:]
        )
        global_max_pooling1d_3 = F.max_pool1d(
            input=conv1d_3_activation, kernel_size=conv1d_3_activation.size()[2:]
        )

        global_max_pooling1d_1_flatten = global_max_pooling1d_1.view(
            global_max_pooling1d_1.size(0), -1
        )
        global_max_pooling1d_2_flatten = global_max_pooling1d_2.view(
            global_max_pooling1d_2.size(0), -1
        )
        global_max_pooling1d_3_flatten = global_max_pooling1d_3.view(
            global_max_pooling1d_3.size(0), -1
        )

        concat_1 = torch.concat([global_max_pooling1d_1_flatten, global_max_pooling1d_2_flatten, global_max_pooling1d_3_flatten], dim=1)
        print(concat_1.shape)
        dense_1 = self.dense_1(concat_1)
        dense_1_activation = torch.relu(dense_1)
        drop_1 = self.drop_1(dense_1_activation)
        dense_2 = self.dense_2(drop_1)
        y = F.sigmoid(dense_2)
        return y

    def embedding_matrix(self):
        return self.embedding_1.weight
