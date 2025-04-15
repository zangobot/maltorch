import torch
from secmlt.models.data_processing.data_processing import DataProcessing
from torch.nn.functional import sigmoid


class MajorityVotingPostprocessing(DataProcessing):
    def __init__(self, threshold: float = 0.5, apply_sigmoid:bool=False):
        super().__init__()
        self.threshold = threshold
        self.apply_sigmoid = apply_sigmoid

    def _process(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.apply_sigmoid:
            x = sigmoid(x)
        x = x.squeeze()
        y_preds = torch.where(x >= self.threshold, 1, 0)
        num_benign = torch.sum(y_preds == 0)
        num_malicious = torch.sum(y_preds == 1)
        return num_malicious/(num_benign+num_malicious)

    def invert(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Apply the inverted transform (if defined).

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            The samples in the input space before the transformation.
        """
        raise NotImplementedError
