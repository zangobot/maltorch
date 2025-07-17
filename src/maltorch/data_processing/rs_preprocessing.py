import torch
from secmlt.models.data_processing.data_processing import DataProcessing
import torch.nn.functional as F


class RandomizedAblationPreprocessing(DataProcessing):
    """
    Daniel Gibert, Giulio Zizzo, Quan Le
    Towards a Practical Defense Against Adversarial Attacks on Deep Learning-Based Malware Detectors via Randomized
    Smoothing.
    ESORICS Workshops, SECAI 2024
    """
    def __init__(self, pabl: float = 0.20, num_versions: int = 100, padding_idx: int = 256, min_len: int = None,
                 max_len: int = None):
        super().__init__()
        self.pabl = pabl
        self.num_versions = num_versions
        self.padding_idx = padding_idx
        self.min_len = min_len,
        self.max_len = max_len


    def _conform_input_size(self, x: torch.Tensor, padding: int = 256) -> torch.Tensor:
        if self.max_len is None and self.min_len is None:
            return x
        batch_size, current_size = x.shape
        if self.min_len is not None:
            padding_needed = max(0, self.min_len - current_size)
            x = F.pad(x, (0, padding_needed), "constant", padding)
        x = x[:, :self.max_len]
        return x

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conform_input_size(x)
        x = x.squeeze()  # Remove all dimensions equal to 1
        vecs = []
        for i in range(self.num_versions):
            mask = torch.rand(x.shape[0]) <= self.pabl
            mask = mask.to(x.device)
            masked_x = x.masked_fill(mask, self.padding_idx)
            vecs.append(masked_x)
        x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True, padding_value=self.padding_idx)
        return x

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

