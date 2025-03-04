import torch
from secmlt.models.data_processing.data_processing import DataProcessing


class MajorityVotingPostprocessing(DataProcessing):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze()
        y_preds = torch.where(x >= self.threshold, 1, 0)
        num_benign = torch.sum(y_preds == 0)
        num_malicious = torch.sum(y_preds == 1)
        return num_malicious/(num_benign+num_malicious)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
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
