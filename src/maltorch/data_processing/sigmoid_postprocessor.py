import torch
from secmlt.models.data_processing.data_processing import DataProcessing


class SigmoidPostprocessor(DataProcessing):
    def _process(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.sigmoid(x)

    def invert(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass