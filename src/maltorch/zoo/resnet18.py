from typing import Optional, Union
import torch.nn.functional as F
from secmlt.models.base_trainer import BaseTrainer
from secmlt.models.data_processing.data_processing import DataProcessing
from torchvision.models import resnet18
from maltorch.zoo.model import PytorchModel, BaseGrayscalePytorchClassifier


class ResNet18(PytorchModel):
    def __init__(self, threshold: float = 0.5):
        super(ResNet18, self).__init__(
            name="ResNet18", gdrive_id="ModelWeightsNotUploadedYet"
        )
        self.model = resnet18(num_classes=1)
        self.threshold = threshold

    def forward(self, x):
        y = self.model(x)
        y = F.sigmoid(y)
        return y


