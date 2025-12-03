from secmlt.models.data_processing.data_processing import DataProcessing
import torch
import thrember


class THREMBERPreprocessing(DataProcessing):
    """
    Joyce, R. J., Miller, G., Roth, P., Zak, R., Zaresky-Williams, E., Anderson, H., ... & Holt, J. (2025).
    EMBER2024--A Benchmark Dataset for Holistic Evaluation of Malware Classifiers.
    arXiv preprint arXiv:2506.05074.
    """

    def invert(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Not supported.")

    def _process(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        extractor = thrember.features.PEFeatureExtractor()
        n_samples = x.shape[0]
        feat_x = torch.zeros((x.shape[0], extractor.dim))
        for i in range(n_samples):
            x_i = x[i, :]
            x_i = x_i.type(torch.int).flatten().tolist()
            if 256 in x_i:
                x_i = x_i[:x_i.index(256)]
            x_bytes = bytearray(x_i)
            feat_x[i, :] = torch.Tensor(extractor.feature_vector(x_bytes))
        return feat_x
