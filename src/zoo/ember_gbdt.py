import ember.features
import lightgbm
import numpy as np
import torch

from src.zoo.model import Model


class EmberGBDT(Model):
    # TODO add gdrive ID
    def __init__(self):
        super().__init__("ember_gbdt", "1MGR7l5c3XSH2dTj2oeefBlKig0bvH2_Z")
        self.tree_model: lightgbm.Booster = lightgbm.Booster()

    def load_pretrained_model(self, device="cpu", model_path=None):
        self._fetch_pretrained_model()
        self.tree_model = lightgbm.Booster(model_file=self.model_path)

    def forward(self, x: torch.Tensor):
        extractor = ember.features.PEFeatureExtractor(print_feature_warning=False)
        n_samples = x.shape[0]
        feat_x = np.zeros((x.shape[0], extractor.dim))
        for i in range(n_samples):
            x_i = x[i, :]
            x_bytes = bytes(x_i.type(torch.int).flatten().tolist())
            feat_x[i, :] = torch.Tensor(extractor.raw_features(x_bytes))
        probabilities = self.tree_model.predict(feat_x, is_reshape=True)
        probabilities = torch.Tensor(probabilities)
        return probabilities
