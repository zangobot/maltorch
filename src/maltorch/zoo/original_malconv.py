"""
Malware Detection by Eating a Whole EXE
Edward Raff, Jon Barker, Jared Sylvester, Robert Brandon, Bryan Catanzaro, Charles Nicholas
https://arxiv.org/abs/1710.09435
This specific model has been trained by TODO.
"""
from typing import Optional, Union

import torch
import torch.nn.functional as F
from secmlt.models.base_trainer import BaseTrainer
from secmlt.models.data_processing.data_processing import DataProcessing
from torch import nn

from maltorch.data_processing.e2e_preprocessor import PaddingPreprocessing
from maltorch.zoo.model import EmbeddingModel, BaseEmbeddingPytorchClassifier


class OriginalMalConv(EmbeddingModel):
    """
    Architecture implementation.
    """
    DEFAULT_MAX_LENGTH = 2**20

    def __init__(self, embedding_size: int = 8,
                 max_len: int =DEFAULT_MAX_LENGTH,
                 threshold: float =0.5,
                 ):
        #https://drive.google.com/file/d/1sPI-gKNAX2hlS-jjepoOu8kwcXZl4VaM/view?usp=drive_link
        super(OriginalMalConv, self).__init__(
            name="Original", gdrive_id="1sPI-gKNAX2hlS-jjepoOu8kwcXZl4VaM", max_len=max_len
        )
        kernel_size = 500
        stride = 500
        out_channels = 128
        output_size=1
        self.embedding_1 = nn.Embedding(num_embeddings=257, embedding_dim=embedding_size, padding_idx=256)
        self.conv1d_1 = nn.Conv1d(in_channels=embedding_size, out_channels=out_channels, kernel_size=(kernel_size,),
                                  stride=(stride,),
                                  groups=1, bias=True)
        self.conv1d_2 = nn.Conv1d(in_channels=embedding_size, out_channels=out_channels, kernel_size=(kernel_size,),
                                  stride=(stride,),
                                  groups=1, bias=True)
        self.dense_1 = nn.Linear(in_features=out_channels, out_features=out_channels, bias=True)
        self.dense_2 = nn.Linear(in_features=out_channels, out_features=output_size, bias=True)
        self.embedding_size = (embedding_size,)
        self.max_len = max_len
        self.threshold = threshold

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
        if preprocessing is None:
            preprocessing = PaddingPreprocessing(max_len=OriginalMalConv.DEFAULT_MAX_LENGTH)
        net = cls(**kwargs)
        net.load_pretrained_model(device=device, model_path=model_path)
        net = net.to(device)  # Explicitly load model to device
        net = net.eval()
        net = BaseEmbeddingPytorchClassifier(
            model=net,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            trainer=trainer,
            threshold=threshold,
        )
        return net
