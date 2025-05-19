import torch
from secmlt.models.data_processing.data_processing import DataProcessing
import math

class KPartitionDeRandomizedPreprocessing(DataProcessing):
    """
    Shoumik Saha, Weinxio Wang, Yigitcan Kaya, Soheil Feizi, Tudo Dumitras
    DRSM: De-Randomized Smoothing on Malware Classifier Providing Certified Robustness
    ICRL'24
    """
    def __init__(self, num_chunks: int = 4, min_chunk_size: int = 500, padding_idx: int = 256):
        super().__init__()
        self.num_chunks = num_chunks
        self.min_chunk_size = min_chunk_size
        self.padding_idx = padding_idx

    def _process(self, x: torch.Tensor) -> torch.Tensor:
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
