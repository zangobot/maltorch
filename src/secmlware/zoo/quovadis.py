"""
Quo Vadis: Hybrid Machine Learning Meta-Model Based on Contextual and Behavioral Malware Representations
Dmitrijs Trizna, ACM AISec 2022
https://arxiv.org/abs/2208.12248
Reimplemented from: https://github.com/dtrizna/quo.vadis/blob/main/models.py
"""

import importlib.util
from typing import List, Union, Optional

from secmlware.zoo.model import EmbeddingModel
from secmlware.zoo.malconv import MalConv

import torch
import torch.nn as nn
import torch.nn.functional as F

from secmlt.models.base_trainer import BaseTrainer
from secmlt.models.data_processing.data_processing import DataProcessing

from secmlware.zoo.model import EmbeddingModel, BaseEmbeddingPytorchClassifier


class MLP(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        hidden_neurons: List[int] = None, 
        batch_norm: bool = False, 
        dropout: float = 0.0
    ):
        super().__init__()
        layers = []
        in_size = input_size

        # If hidden_neurons is None or empty, create a single layer mapping input to output
        if not hidden_neurons:
            layers.append(nn.Linear(in_size, output_size))
        else:
            # Add hidden layers
            for h in hidden_neurons:
                layers.append(nn.Linear(in_size, h))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_size = h
            # Add output layer
            layers.append(nn.Linear(in_size, output_size))

        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor):
        return self.net(x)


class Core1DConvNet(EmbeddingModel):
    def __init__(
        self,
        name: str,
        gdrive_id: str,
        vocab_size: int = 152,
        embedding_size: int = 96,
        filter_sizes: List[int] = [2, 3, 4, 5],
        num_filters: List[int] = [128, 128, 128, 128],
        batch_norm_conv: bool = False,
        hidden_neurons: List[int] = [128],
        batch_norm_ffnn: bool = False,
        dropout: float = 0.5,
        output_dim: int = 128
    ):
        super().__init__(name, gdrive_id)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=0
        )

        self.conv1d_modules = nn.ModuleList()
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            layers = [nn.Conv1d(
                in_channels=embedding_size,
                out_channels=num_filter,
                kernel_size=filter_size
            )]
            if batch_norm_conv:
                layers.append(nn.BatchNorm1d(num_filter))
            self.conv1d_modules.append(nn.Sequential(*layers))

        conv_output_size = sum(num_filters)

        self.mlp = MLP(
            input_size=conv_output_size,
            output_size=output_dim,
            hidden_neurons=hidden_neurons,
            batch_norm=batch_norm_ffnn,
            dropout=dropout
        )

    def embedding_layer(self):
        return self.embedding
    
    def embedding_matrix(self):
        return self.embedding.weight

    def embed(self, x: torch.Tensor):
        emb_x = self.embedding(x)
        emb_x = emb_x.permute(0, 2, 1)
        return emb_x
    
    def _forward_embed_x(self, x: torch.Tensor):
        conv_outputs = [
            self.conv_and_max_pool(x, conv_module)
            for conv_module in self.conv1d_modules
        ]
        x = torch.cat(conv_outputs, dim=1)  # [batch_size, conv_output_size]
        out = self.mlp(x)
        return out
     
    @staticmethod
    def conv_and_max_pool(x, conv_module: nn.Module):
        """Apply convolution and global max pooling."""
        x = conv_module(x)
        x = F.relu(x)
        x = x.max(dim=2)[0]  # Global max pooling over the sequence length
        return x

    def forward(self, inputs: torch.Tensor):
        # inputs: [batch_size, sequence_length]
        embedded = self.embedding(inputs)  # [batch_size, sequence_length, embedding_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, sequence_length]
        
        # Apply convolution and pooling
        conv_outputs = [
            self.conv_and_max_pool(embedded, conv_module)
            for conv_module in self.conv1d_modules
        ]
        x = torch.cat(conv_outputs, dim=1)  # [batch_size, conv_output_size]
        out = self.mlp(x)
        return out


class Emulation(Core1DConvNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        speakeasy_spec = importlib.util.find_spec("speakeasy")
        if speakeasy_spec is None:
            raise ImportError("[-] 'speakeasy' is required for emulation module. Use 'pip install speakeasy' to install.")
        from secmlware.data_processing.dynamic.emulation import PEDynamicFeatureExtractor
        from secmlware.data_processing.dynamic.tokenization import JSONTokenizerNaive

        self.dynamic_extractor = PEDynamicFeatureExtractor(speakeasy_record_fields=['apis.api_name'], record_limits=None)
        
        # TODO: right now reusing APIs from Quo.Vadis paper
        # should be based on data in training set for fair comparison
        # so use: self.tokenizer.train(corpus, vocab_size)
        self.tokenizer = JSONTokenizerNaive(vocab="quovadis") 
 
    def emulate(self, pe: Union[str, bytes]):
        return self.dynamic_extractor.emulate(pe)
    
    def encode(self, report: dict):
        return self.tokenizer.encode(report)
    
    def preprocess(self, pe: Union[str, bytes]):
        report = self.emulate(pe)
        return self.encode(report)

    def forward(self, x: torch.Tensor):
        x = self.embed(x)
        output = self._forward_embed_x(x)
        return output

        
class QuoVadis(EmbeddingModel):
    def __init__(
            self,
            embedding_size: int = 8,
            padding_value: int = 256,
            representation_size: int = 128,
            quo_vadis_hidden_neurons: List[int] = [128],
            modules: List[str] = ['malconv', 'emulation'],
    ):
        super(QuoVadis, self).__init__(
            name="QuoVadis", gdrive_id="NotYetPreTrained"
        )
        self.module_set = {}
        if 'emulation' in modules:
            self.module_set['emulation'] = Emulation(
                name="Emulation",
                gdrive_id="NotYetPreTrained",
                embedding_size=embedding_size,
                output_dim=representation_size
            )
        if 'malconv' in modules:
            self.module_set['malconv'] = MalConv(
                embedding_size=embedding_size,
                padding_value=padding_value,
                out_size=representation_size
            )
        
        self.representation_size_total = representation_size * len(self.module_set)

        self.quovadis_meta_model = MLP(
            input_size=self.representation_size_total,
            output_size=1,
            hidden_neurons=quo_vadis_hidden_neurons,
            batch_norm=False,
            dropout=0.5
        )

    @classmethod
    def create_model(
            cls,
            model_path: str = None,
            device: str = "cpu",
            preprocessing: DataProcessing = None,
            postprocessing: DataProcessing = None,
            trainer: BaseTrainer = None,
            threshold: Optional[float] = 0.5,
            **kwargs,
    ) -> BaseEmbeddingPytorchClassifier:
        net = cls(**kwargs)
        net.load_pretrained_model(device=device, model_path=model_path)
        net.eval()

        for module in net.module_set.values():
            module.load_pretrained_model(device=device, model_path=model_path)
            module.eval()

        net = BaseEmbeddingPytorchClassifier(
            model=net,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            trainer=trainer,
            threshold=threshold,
        )
        return net
    
    # TODO: consider if embed in QuoVadis should be a torch.cat() output from early modules
    # instead of using the embed() method of each module
    def embed(self, x: torch.Tensor):
        embeddings = []
        for module in self.module_set.values():
            embeddings.append(module.embed(x))
        return torch.cat(embeddings, dim=1)
    
    def embedding_layer(self):
        raise NotImplementedError("QuoVadis does not have a single embedding layer")
    
    def embedding_matrix(self):
        raise NotImplementedError("QuoVadis does not have a single embedding matrix")

    def _forward_embed_x(self, x: torch.Tensor):
        representations = []
        for module in self.module_set.values():
            representations.append(module._forward_embed_x(x))
        return torch.cat(representations, dim=1)
    
    def forward(self, x: torch.Tensor):
        x = self.embed(x)
        output = self._forward_embed_x(x)
        return self.quovadis_meta_model(output)
