from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
from secmlt.models.base_model import BaseModel
from secmlt.models.base_trainer import BaseTrainer
from secmlt.models.data_processing.data_processing import DataProcessing
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from maltorch.utils.config import Config
from maltorch.utils.utils import download_gdrive


class Model(torch.nn.Module, ABC):
    def __init__(self, name: str, gdrive_id: Optional[str]):
        super().__init__()
        self.name = name
        self.gdrive_id = gdrive_id
        self.model_path = Config.MODEL_ZOO_FOLDER / self.name

    def _fetch_pretrained_model(self):
        if not Config.MODEL_ZOO_FOLDER.exists():
            Config.MODEL_ZOO_FOLDER.mkdir()
        if not self.model_path.exists():
            if self.gdrive_id is not None:
                download_gdrive(gdrive_id=self.gdrive_id, fname_save=self.model_path)

    def load_pretrained_model(self, device="cpu", model_path=None): ...

    @classmethod
    def create_model(
        cls,
        model_path: Optional[str] = None,
        device: str = "cpu",
        preprocessing: DataProcessing = None,
        postprocessing: DataProcessing = None,
        trainer: BaseTrainer = None,
    ) -> BaseModel: ...


class PytorchModel(Model):
    def load_pretrained_model(self, device="cpu", model_path=None):
        path = self.model_path
        if model_path is None:
            self._fetch_pretrained_model()
        else:
            path = model_path
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)

    @classmethod
    def create_model(
        cls,
        model_path: Optional[str] = None,
        device: str = "cpu",
        preprocessing: DataProcessing = None,
        postprocessing: DataProcessing = None,
        trainer: BaseTrainer = None,
        **kwargs,
    ) -> BaseModel:
        net = cls(**kwargs)
        net.load_pretrained_model(device=device, model_path=model_path)
        net = net.to(device) # Explicitly load model to device
        net.eval()
        net = BasePytorchClassifier(
            model=net,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            trainer=trainer,
        )
        return net


class BaseEmbeddingPytorchClassifier(BasePytorchClassifier):
    def __init__(
        self,
        model: torch.nn.Module,
        preprocessing: DataProcessing = None,
        postprocessing: DataProcessing = None,
        trainer: BaseTrainer = None,
        threshold: Optional[Union[float, None]] = 0.5,
    ):
        super().__init__(model, preprocessing, postprocessing, trainer)
        self.threshold = threshold

    def embed(self, x: torch.Tensor):
        return self.model.embed(x)

    def embedding_matrix(self):
        return self.model.embedding_matrix()

    def embedding_layer(self):
        return self.model.embedding_layer()

    def predict(self, x: torch.Tensor):
        if self.threshold is None:
            return super().predict(x)
        scores = self.decision_function(x)
        labels = (scores > self.threshold).int()
        return labels


class EmbeddingModel(PytorchModel, ABC):
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
        net = net.to(device) # Explicitly load model to device
        net.eval()
        net = BaseEmbeddingPytorchClassifier(
            model=net,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            trainer=trainer,
            threshold=threshold,
        )
        return net

    def __init__(
        self, name: str, gdrive_id: Optional[str], input_embedding: bool = False
    ):
        super().__init__(name, gdrive_id)
        self.input_embedding = input_embedding

    @abstractmethod
    def embed(self, x):
        pass

    @abstractmethod
    def embedding_layer(self):
        pass

    @abstractmethod
    def embedding_matrix(self):
        pass

    @abstractmethod
    def _forward_embed_x(self, x):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            the sample to test
        Returns
        -------
        torch.Tensor
            the result of the forward pass
        """
        x = self.embed(x)
        output = self._forward_embed_x(x)
        return output


class BaseGrayscalePytorchClassifier(BasePytorchClassifier):
    def __init__(
        self,
        model: torch.nn.Module,
        preprocessing: DataProcessing = None,
        postprocessing: DataProcessing = None,
        trainer: BaseTrainer = None,
        threshold: Optional[Union[float, None]] = 0.5,
    ):
        super().__init__(model, preprocessing, postprocessing, trainer)
        self.threshold = threshold

    def predict(self, x: torch.Tensor):
        if self.threshold is None:
            return super().predict(x)
        scores = self.decision_function(x)
        labels = (scores > self.threshold).int()
        return labels
