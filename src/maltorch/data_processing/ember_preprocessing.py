from secmlt.models.data_processing.data_processing import DataProcessing
import torch
import ember


class EMBERPreprocessing(DataProcessing):
    """
    Anderson, H. S., & Roth, P. (2018).
    EMBER: an open dataset for training static pe malware machine learning models. 
    arXiv preprint arXiv:1804.04637.
    """

    def invert(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Not supported.")

    def _process(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        extractor = ember.features.PEFeatureExtractor(print_feature_warning=False)
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
