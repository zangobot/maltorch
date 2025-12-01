from enum import Enum
from typing import Optional, overload

import torch
import requests

from secmlt.models.base_model import BaseModel
from secmlt.models.base_trainer import BaseTrainer
from secmlt.models.data_processing.data_processing import DataProcessing
from sqlalchemy.util import OrderedDict
from torch.utils.data import DataLoader

from maltorch.utils.utils import compute_hash_from_sample
from maltorch.zoo.model import Model


class VTEngineLabel(Enum):
    AVAST = "avast!"
    AVG = "avg"
    BITDEFENDER = "bitdefender"
    CLAMAV = "clamav"
    CROWDSTRIKE = "crowdstrike"
    DRWEB = "drweb"
    ESET_NOD32 = "eset-nod32"
    F_SECURE = "f-secure"
    KASPERSKY = "kaspersky"
    MALWAREBYTES = "malwarebytes"
    MICROSOFT_DEFENDER = "windows-defender"
    PANDA = "panda"
    SOPHOS = "sophos"
    SYMANTEC_NORTON = "symantec-norton"
    TREND_MICRO = "trendmicro-housecall"
    VIRUSBUSTER = "virusbuster"


class VirusTotalBaseModel(BaseModel):

    def __init__(self,
                 api_key: str,
                 av_list: list[str] = None,
                 threshold: int = 5,
                 preprocessing: DataProcessing = None,
                 postprocessing: DataProcessing = None,
                 **kwargs):
        super().__init__(preprocessing=preprocessing, postprocessing=postprocessing)
        self.api_key = api_key
        self._filter_avs = av_list
        self.threshold = threshold

    def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        y = self._decision_function(x)
        predictions = y.sum(dim=-1) > self.threshold
        return predictions


    def _decision_function(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        y = torch.zeros((x.shape[0], len(VTEngineLabel) if self._filter_avs is None else len(self._filter_avs)))
        for i in range(x.shape[0]):
            hash_xi = compute_hash_from_sample(x[i, :])
            base_url = "https://www.virustotal.com/api/v3/"
            headers = {"x-apikey": self.api_key}
            response = requests.get(f"{base_url}files/{hash_xi}",
                                    headers=headers)
            av_responses = None
            if response.status_code == 200:
                av_responses = response.json()["data"]["scans"]
                if self._filter_avs is not None:
                    av_responses = OrderedDict(
                        [(k, av_responses[k]['detected'] == True) for k in av_responses if k in self._filter_avs])
            else:
                print(f"Error fetching data: {response.status_code}")
            y[i, :] = torch.Tensor(av_responses.values())
        return y

    def gradient(self, x: torch.Tensor, y: int, *args, **kwargs) -> torch.Tensor:
        pass

    def train(self, dataloader: DataLoader) -> "BaseModel":
        pass


class VirusTotal(Model):
    def __init__(self, name: str, api_key: str, gdrive_id: Optional[str]):
        super().__init__(name, None)
        self.api_key = api_key

    def load_pretrained_model(self, device="cpu", model_path=None):
        return None

    @classmethod
    def create_model(
            cls,
            api_key: str,
            preprocessing: DataProcessing = None,
            postprocessing: DataProcessing = None,
            trainer: BaseTrainer = None,
            **kwargs
    ) -> BaseModel:
        ...
