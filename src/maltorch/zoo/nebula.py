"""
Nebula: Self-Attention for Dynamic Malware Analysis
Dmitrijs Trizna; Luca Demetrio; Battista Biggio; Fabio Roli
IEEE Transactions on Information Forensics andSecurity (TIFS), 2024
https://ieeexplore.ieee.org/document/10551436
Reimplemented from: https://github.com/dtrizna/nebula/blob/main/nebula/__init__.py
"""

import math
import importlib.util
from typing import Union, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from maltorch.zoo.model import EmbeddingModel


class PositionalEncoding(nn.Module):
    """ From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoderModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,  # size of vocabulary
            maxlen: int,  # maximum length of input sequence
            dModel: int = 32,  # embedding & transformer dimension
            nHeads: int = 8,  # number of heads in nn.MultiheadAttention
            dHidden: int = 200,  # dimension of the feedforward network model in nn.TransformerEncoder
            nLayers: int = 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            numClasses: int = 1,  # 1 ==> binary classification
            classifier_head: Optional[list] = [64],  # decoder's classifier FFNN complexity
            pretrain_layers: Optional[List] = None,
            layerNorm: bool = False,  # whether to normalize decoder's FFNN layers
            norm_first: bool = True,  # whether to normalize before or after FFNN layers
            dropout: float = 0.3,
            pooling: str = "mean",
            skip_embedding: bool = False,
            causal_attention: bool = False
    ):
        super().__init__()
        self.__name__ = 'TransformerEncoderModel'
        
        assert dModel % nHeads == 0, "nheads must divide evenly into d_model"
        assert pooling in ["mean", "flatten", "cls", None]

        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.skip_embedding = skip_embedding
        self.encoder = nn.Embedding(vocab_size, dModel)
        self.pos_encoder = PositionalEncoding(dModel, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model=dModel,
            nhead=nHeads,
            dim_feedforward=dHidden,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nLayers)
        self.d_model = dModel
        self.layerNorm = layerNorm
        self.dropout = dropout
        
        self.pooling_type = pooling
        if pooling == "mean" or pooling is None:
            input_neurons = self.d_model
        elif pooling == "flatten":
            input_neurons = int(self.maxlen * dModel)
        elif pooling == "cls":
            raise NotImplementedError
        
        self.causal_attention = causal_attention
        if self.causal_attention: # override pooling settings if causal
            input_neurons = self.d_model
            self.pooling_type = None
        
        self.classifier_head_nr_list = classifier_head
        self.num_classes_out = numClasses
        self.classifier_head = None
        if self.classifier_head_nr_list is not None:
            # model is initiated as a downstream model -- setup classifier head
            self.classifier_head_layers = []
            if len(self.classifier_head_nr_list) > 0:
                self.classifier_head_layers = self._build_ffnn_layers(
                    start_neurons=input_neurons,
                    hidden_layers=self.classifier_head_nr_list
                )
            # if numClasses is None, then classifier outputs last value from classified_head
            if self.num_classes_out is not None:
                self.final_layer_in = classifier_head[-1] if len(classifier_head) > 0 else input_neurons
                self.final_layer_out =  1 if self.num_classes_out == 2 else self.num_classes_out
                self.classifier_head_layers.append(nn.Linear(self.final_layer_in, self.final_layer_out))
            # join in a single classifier head
            self.classifier_head = nn.Sequential(*self.classifier_head_layers)
            
        self.pretrain_layers = None
        if pretrain_layers is not None:
            self.pretrain_layers = []
            if len(pretrain_layers) > 0:
                self.pretrain_layers = self._build_ffnn_layers(
                    start_neurons=input_neurons,
                    hidden_layers=pretrain_layers
                )
            final_layer_in = pretrain_layers[-1] if len(pretrain_layers) > 0 else input_neurons
            # NOTE: bias removed in last layer as in https://github.com/karpathy/nanoGPT/blob/master/model.py#L133
            self.pretrain_layers.append(nn.Linear(final_layer_in, self.vocab_size, bias=False))
            self.pretrain_layers = nn.Sequential(*self.pretrain_layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # NOTE: from: https://github.com/karpathy/nanoGPT/blob/master/model.py#L162
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_ffnn_layers(self, start_neurons: int, hidden_layers: list) -> nn.Sequential:
        ffnn = []
        if hidden_layers is None:
            return ffnn
        for i, h in enumerate(hidden_layers):
            ffnnBlock = []
            if i == 0:
                ffnnBlock.append(nn.Linear(start_neurons, h))
            else:
                ffnnBlock.append(nn.Linear(hidden_layers[i - 1], h))

            # add LayerNorm to every layer except last
            if self.layerNorm and i < len(hidden_layers) - 1:
                ffnnBlock.append(nn.LayerNorm(h))

            ffnnBlock.append(nn.ReLU())
            ffnnBlock.append(nn.Dropout(self.dropout))

            ffnn.append(nn.Sequential(*ffnnBlock))
        return ffnn
    
    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def embed(self, x: Tensor) -> Tensor:
        encoded_x = self.encoder(x) * math.sqrt(self.d_model)
        encoded = self.pos_encoder(encoded_x)
        return encoded

    def pooling(self, x: Tensor) -> Tensor:
        if self.causal_attention or self.pooling_type is None:
            return x
        if self.pooling_type == "flatten":
            x = x.view(x.size(0), -1)
        if self.pooling_type == "mean":
            x = torch.mean(x, dim=1)
        return x

    def core(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        x = src if self.skip_embedding else self.embed(src)
        if self.causal_attention:
            src_mask = self._generate_square_subsequent_mask(x.shape[1]).to(x.device)
        x = self.transformer_encoder(x, src_mask, is_causal=self.causal_attention)
        x = self.pooling(x)
        return x

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        x = self.core(x, src_mask)
        if self.classifier_head is not None:
            x = self.classifier_head(x)
        return x

    def pretrain(self, x: Tensor) -> Tensor:
        x = self.core(x)
        if self.pretrain_layers is not None:
            x = self.pretrain_layers(x)
        return x


class TransformerEncoderChunks(TransformerEncoderModel):
    """
    Slices global attention to multiple separate chunks, which are processed independently till classifier head.
    Main model used in: https://arxiv.org/abs/2310.10664
    'Chunk' term borrowed from longformer:
    https://github.com/allenai/longformer/blob/master/longformer/sliding_chunks.py
    """

    def __init__(self, chunk_size: int = 64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name__ = 'TransformerEncoderChunks'

        self.chunk_size = chunk_size
        self.nr_of_chunks = self.maxlen / self.chunk_size
        if self.nr_of_chunks != int(self.nr_of_chunks):
            self.nr_of_chunks = int(self.nr_of_chunks) + 1
        
    def split(self, src: Tensor) -> List[Tensor]:
        chunks = []
        for chunk in torch.split(src, split_size_or_sections=self.chunk_size, dim=1):
            if chunk.shape[1] < self.chunk_size:
                pad_mask = (0, self.chunk_size - chunk.shape[1])
                chunk = F.pad(chunk, pad=pad_mask)
            chunks.append(chunk)
        return chunks

    def embed(self, chunks: List[Tensor]) -> List[Tensor]:
        encoded_chunks = []
        for chunk in chunks:
            encoded_chunk = self.encoder(chunk) * math.sqrt(self.d_model)
            encoded_chunk = self.pos_encoder(encoded_chunk)
            encoded_chunks.append(encoded_chunk)
        return encoded_chunks

    def transform(self, chunks: List[Tensor], src_mask: Optional[Tensor] = None) -> List[Tensor]:
        transformed_chunks = []
        for chunk in chunks:
            transformed_chunk = self.transformer_encoder(chunk, src_mask)
            transformed_chunks.append(transformed_chunk)
        return transformed_chunks

    def core(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        chunks = self.split(src)
        chunks = self.embed(chunks)  # [(batch_size, chunk_size, d_model), ..]
        chunks = self.transform(chunks, src_mask)  # [(batch_size, chunk_size, d_model), ..]
        x = torch.cat(chunks, dim=1)
        # NOTE: after .cat() shape is: (batch_size, nr_of_chunks * chunk_size, d_model)
        # where nr_of_chunks = int(maxlen / self.chunk_size) + 1
        x = self.pooling(x)
        return x


class Nebula(EmbeddingModel):
    def __init__(
            self,
            vocab_size: int = 50000,
            seq_len: int = 512,
            tokenizer: str = 'whitespace',
    ):
        # optional library imports
        speakeasy_spec = importlib.util.find_spec("speakeasy")
        if speakeasy_spec is None:
            raise ImportError("[-] 'speakeasy' is required for emulation module. Use 'pip install speakeasy' to install.")
        from maltorch.data_processing.dynamic.emulation import PEDynamicFeatureExtractor
        
        nltk_spec = importlib.util.find_spec("nltk")
        if nltk_spec is None:
            raise ImportError("[-] 'nltk' is required for tokenization module. Use 'pip install nltk' to install.")
        from maltorch.data_processing.dynamic.tokenization import JSONTokenizerNaive, JSONTokenizerBPE

        self.vocab_size = vocab_size
        self.seq_len = seq_len

        # dynamic extractor setup
        self.dynamic_extractor = PEDynamicFeatureExtractor() 
        
        # tokenizer initialization
        assert tokenizer in ['bpe', 'whitespace'], "tokenizer must be in ['bpe', 'whitespace']"
        if tokenizer == 'bpe':
            self.tokenizer = JSONTokenizerBPE(
                vocab_size=self.vocab_size,
                seq_len=self.seq_len,
            )
        if tokenizer == 'whitespace':
            self.tokenizer = JSONTokenizerNaive(
                vocab_size=self.vocab_size,
                seq_len=self.seq_len,
            )
        
        # PyTorch model initialization
        torch_model_config = {
            "vocab_size": self.vocab_size,
            "maxlen": self.seq_len,
            "chunk_size": 64, # self-attention window size
            "dModel": 64,  # embedding & transformer dimension
            "nHeads": 8,  # number of heads in nn.MultiheadAttention
            "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
            "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            "numClasses": 1, # binary classification
            "classifier_head": [64], # classifier head depth
            "layerNorm": False,
            "dropout": 0.3,
            "norm_first": True,
            "pooling": "flatten"
        }
        self.model = TransformerEncoderChunks(**torch_model_config)

    def dynamic_analysis_pe_file(self, pe_file: Union[str, bytes]) -> dict:
        if isinstance(pe_file, str):
            with open(pe_file, "rb") as f:
                bytez = f.read()
        elif isinstance(pe_file, bytes):
            bytez = pe_file
        else:
            raise ValueError("preprocess(): data must be a path to a PE file or a bytes object")

        dynamic_features_json = self.dynamic_extractor.emulate(data=bytez)
        return dynamic_features_json

    def preprocess(self, emulation_report: dict) -> Tensor:
        dynamic_features = self.tokenizer.encode(emulation_report)
        return torch.Tensor(dynamic_features).long()
    
    def predict_proba(self, dynamic_features: Tensor) -> float:
        dynamic_features = torch.Tensor(dynamic_features).long()
        with torch.no_grad():
            logits = self.model(dynamic_features)
        return torch.sigmoid(logits).item()
    
    def predict_sample(self, pe_file: Union[str, bytes]) -> float:
        dynamic_features_json = self.dynamic_analysis_pe_file(pe_file)
        dynamic_features = self.preprocess(dynamic_features_json)
        return self.predict_proba(dynamic_features)

    def embed(self, x: Tensor) -> Tensor:
        return self.model.embed(x)
    
    def embedding_layer(self):
        return self.model.encoder
    
    def embedding_matrix(self):
        return self.model.encoder.weight
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
