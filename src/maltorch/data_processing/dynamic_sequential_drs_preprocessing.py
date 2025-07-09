import torch
from secmlt.models.data_processing.data_processing import DataProcessing
import math
import torch.nn.functional as F



class DynamicSequentialDeRandomizedPreprocessing(DataProcessing):
    """
    Daniel Gibert, Giulio Zizzo, Quan Le, Jordi Planes
    Adversarial Robustness of Deep Learning-based Malware Detectors via (De) Randomized Smoothing
    IEEE Access 2024 - Sequential Chunks-based (De)Randomized Smoothing
    """
    def __init__(self,
                 file_percentage: float = 0.05,
                 num_chunks: int = 100,
                 padding_idx: int = 256,
                 min_chunk_size: int = 500,
                 min_len: int = None,
                 max_len: int = None
                 ):
        super().__init__()
        self.file_percentage = file_percentage
        self.num_chunks = num_chunks
        self.padding_idx = padding_idx
        self.min_len = min_len,
        self.max_len = max_len,
        self.min_chunk_size = min_chunk_size


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
        max_size = x.size()[0]
        group_size = math.ceil(max_size * self.file_percentage)
        group_size = max(self.min_chunk_size,
                         group_size)  # The chunk size has to be at least equal to the kernel size of the first convolutional layer
        overlap_size = group_size - int(max_size / self.num_chunks)

        to_substract_A = math.ceil((group_size - overlap_size) / self.num_chunks)
        to_substract_B = math.ceil((group_size - overlap_size) * self.file_percentage)
        to_substract = to_substract_B - to_substract_A

        vecs = []
        for i in range(self.num_chunks):
            start = i * (group_size - overlap_size - to_substract)
            end = start + group_size
            vecs.append(x[start:end])

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

