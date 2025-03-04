import torch
from secmlt.models.data_processing.data_processing import DataProcessing


class RandomizedDeletionPreprocessing(DataProcessing):
    """
    Zhougun Huang, Niel Marchant, Keane Lucas, Lujo Bauer, Olya Ohrimenko, Benjamin I. P. Rubenstein
    RS-Del: Edit Distance Robustness Certificates for Sequence Classifiers via Randomized Deletion
    NeurIPS 2023,
    """
    def __init__(self, pdel: float = 0.03, num_versions: int = 100, padding_idx: int = 256):
        super().__init__()
        self.pdel = pdel
        self.num_versions = num_versions
        self.padding_idx = padding_idx

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze()  # Remove all dimensions equal to 1
        vecs = []
        for i in range(self.num_versions):
            mask = torch.rand(x.shape[0]) > self.pdel
            mask = mask.to(x.device)
            masked_x = torch.masked_select(x, mask=mask)
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

