from typing import Optional
import torch
from secmlt.models.base_model import BaseModel
from secmlt.models.base_trainer import BaseTrainer
from secmlt.models.data_processing.data_processing import DataProcessing
from torch.utils.data import DataLoader
from src.secmlware.zoo.model import Model
from src.secmlware.zoo.sequential_pipeline import SequentialPipelineClassifier, ErrorHandling, _SequentialPipelineModel
from src.secmlware.zoo.ember_gbdt import EmberGBDT
from src.secmlware.zoo.malconv import MalConv
from src.secmlware.zoo.yara_matching import Yara
from pathlib import Path


class SliferClassifier(SequentialPipelineClassifier):
    # def __init__(self, models=None, error_handling: Optional[ErrorHandling] = ErrorHandling.RAISE_ERROR):
    #     if models is None:
    #         models_type = [Yara(), MalConv(), EmberGBDT()]
    #     if len(models) != 3:
    #         raise ValueError("Slifer model must have 3 models.")
    #     # if models[0] is not Yara and models[1] is not models[1].isinstance(MalConv) and models[2] is not EmberGBDT:
    #     #     raise ValueError("Slifer model is composed of YARA, MalConv and EmberGBDT.")
    #     # super().__init__(models, error_handling)

    # initialize a sequential pipeline model with Yara, MalConv and EmberGBDT by default
    @classmethod
    def create_model(
            cls,
            device: str = "cpu",
            error_handling: ErrorHandling = ErrorHandling.RAISE_ERROR,
            preprocessing: DataProcessing = None,
            postprocessing: DataProcessing = None,
            trainer: BaseTrainer = None,
            rules_path: Optional[str] = None,
            gbdt_path: Optional[str] = None
    ) -> BaseModel:
        if rules_path is None:
            rules_path = Path(__file__).resolve().parent / 'models' / 'yara_rules'
        if gbdt_path is None:
            gbdt_path = Path(__file__).resolve().parent / 'models' / 'ember_model.txt'
        return _SequentialPipelineModel(models=[Yara.create_model(rules_path=Path(__file__).resolve().parent / 'models'
                                                                                                        '/yara_rules'),
                                                MalConv().create_model(device='cpu'),
                                                EmberGBDT.create_model(
                                                    model_path=Path(__file__).resolve().parent / 'models' / 'ember_model.txt')],
                                        error_handling=error_handling)
