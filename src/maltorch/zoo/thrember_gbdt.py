import abc
from typing import Optional

import lightgbm
from secmlt.models.base_model import BaseModel
from secmlt.models.base_trainer import BaseTrainer
from secmlt.models.data_processing.data_processing import DataProcessing

from maltorch.data_processing.thrember_preprocessing import THREMBERPreprocessing
from maltorch.zoo.gbdt import GBDTModel
from maltorch.zoo.model import Model



class ThremberGBDT(Model):
    def __init__(self, model_path: Optional[str] = None):
        # https://drive.google.com/file/d/1UUYwTT9ST_0Cs_854aVHXUOfgSVXACaD/view?usp=drive_link
        super().__init__(
            name="ember2024_gbdt", gdrive_id="1UUYwTT9ST_0Cs_854aVHXUOfgSVXACaD"
        )
        self.tree_model = None
        self.load_pretrained_model(model_path=model_path)

    def load_pretrained_model(self, device: str = "cpu", model_path: str = None):
        if model_path is None:
            self._fetch_pretrained_model()
            self.tree_model = lightgbm.Booster(model_file=self.model_path)
        else:
            self.tree_model = lightgbm.Booster(model_file=model_path)

    @classmethod
    def create_model(
            cls,
            model_path: Optional[str] = None,
            threshold: int = 0.5,
            preprocessing: DataProcessing = None,
            postprocessing: DataProcessing = None,
            trainer: BaseTrainer = None,
    ) -> BaseModel:
        model = cls(model_path=model_path)
        return GBDTModel(tree_model=model.tree_model, threshold=threshold, preprocessing=THREMBERPreprocessing())
