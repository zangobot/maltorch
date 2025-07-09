import torch
from secmlt.models.data_processing.data_processing import DataProcessing
import math
import torch.nn.functional as F


class KPartitionDeRandomizedPreprocessing(DataProcessing):
    """
    Shoumik Saha, Weinxio Wang, Yigitcan Kaya, Soheil Feizi, Tudo Dumitras
    DRSM: De-Randomized Smoothing on Malware Classifier Providing Certified Robustness
    ICRL'24
    """
    def __init__(self, num_chunks: int = 4, min_chunk_size: int = 500, padding_idx: int = 256, min_len: int = None, max_len: int = None):
        super().__init__()
        self.num_chunks = num_chunks
        self.min_chunk_size = min_chunk_size
        self.padding_idx = padding_idx
        self.min_len = min_len
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
        x = x.squeeze()
        L = len(x)
        chunk_size = math.ceil(L/self.num_chunks)
        if chunk_size < self.min_chunk_size:
            chunk_size = self.min_chunk_size
        x = [x[i:i+chunk_size] for i in range(0, L, chunk_size)]
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
