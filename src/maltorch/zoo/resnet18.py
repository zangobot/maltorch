from torchvision.models import resnet18
from maltorch.zoo.model import PytorchModel


class ResNet18(PytorchModel):
    def __init__(self, threshold: float = 0.5):
        super(ResNet18, self).__init__(
            name="ResNet18", gdrive_id="ModelWeightsNotUploadedYet"
        )
        self.model = resnet18(num_classes=1)
        self.threshold = threshold

    def forward(self, x):
        y = self.model(x)
        return y


