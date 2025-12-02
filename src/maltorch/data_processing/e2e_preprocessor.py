import torch
from secmlt.models.data_processing.data_processing import DataProcessing


class PaddingPreprocessing(DataProcessing):
    def _process(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        padding = torch.nn.ConstantPad1d((0, self.max_len), 256)
        return padding(x)

    def invert(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    def __init__(self, max_len:int=2**20):
        self.max_len = max_len