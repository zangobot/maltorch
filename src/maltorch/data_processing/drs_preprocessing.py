import torch
from secmlt.models.data_processing.data_processing import DataProcessing


class DeRandomizedPreprocessing(DataProcessing):
    """
    Daniel Gibert, Giulio Zizzo, Quan Le
    Certified Robustness of Static Deep Learning-based Malware Detectors against Patch and Append Attacks
    AISec '23

    Shoumik Saha, Weinxio Wang, Yigitcan Kaya, Soheil Feizi, Tudo Dumitras
    DRSM: De-Randomized Smoothing on Malware Classifier Providing Certified Robustness
    ICRL'24
    """
    def __init__(self, chunk_size: int = 512, padding_idx: int = 256):
        super().__init__()
        self.chunk_size = chunk_size
        self.padding_idx = padding_idx

    def _process(self, x: torch.Tensor) -> torch.Tensor:
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

