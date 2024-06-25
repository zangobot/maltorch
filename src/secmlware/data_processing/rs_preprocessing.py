import torch
from secmlt.models.data_processing.data_processing import DataProcessing


class RandomizedAblationPreprocessing(DataProcessing):
    """
    Daniel Gibert, Giulio Zizzo, Quan Le
    Towards a Practical Defense Against Adversarial Attacks on Deep Learning-Based Malware Detectors via Randomized
    Smoothing.
    ESORICS Workshops, SECAI 2024
    """
    def __init__(self, pabl: float = 0.97, num_versions: int = 100, padding_value: int = 256):
        super().__init__()
        self.pabl = pabl
        self.num_versions = num_versions
        self.padding_value = padding_value

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze()  # Remove all dimensions equal to 1
        vecs = []
        for i in range(self.num_versions):
            mask_value_prob = 1.0 - self.pabl
            mask = torch.rand(x.shape[0]) <= mask_value_prob
            masked_x = x.masked_fill(mask, self.padding_value)
            vecs.append(masked_x)
        x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True, padding_value=self.padding_value)
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

