import torch
from secmlt.models.data_processing.data_processing import DataProcessing
import torch.nn.functional as F


class FixedSizeChunkDeRandomizedPreprocessing(DataProcessing):
    """
    Daniel Gibert, Giulio Zizzo, Quan Le
    Certified Robustness of Static Deep Learning-based Malware Detectors against Patch and Append Attacks
    AISec '23
    """
    def __init__(self, chunk_size: int = 512, padding_idx: int = 256, min_len: int = None,
                 max_len: int = None):
        super().__init__()
        self.chunk_size = chunk_size
        self.padding_idx = padding_idx
        self.min_len = min_len,
        self.max_len = max_len,

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
        x = [x[i:i + self.chunk_size] for i in range(0, len(x), self.chunk_size)]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.padding_idx)
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
        return x.view(-1)

