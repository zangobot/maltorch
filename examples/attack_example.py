from pathlib import Path

import torch
from secml2.metrics.classification import Accuracy
from secml2.trackers.trackers import LossTracker
from torch.utils.data import TensorDataset, DataLoader

from secml2malware.adv.evasion.partialdos import PartialDOS
from secml2malware.data.loader import load_from_folder
from secml2malware.zoo.malconv import MalConv

device = "cpu"

folder = Path(__file__).parent
X = load_from_folder(folder, "file")
X.to(device)
pdos_attack = PartialDOS(
    20,
    58,
    trackers=[LossTracker()],
)
malconv = MalConv.create_model(device=device)
y = torch.LongTensor([1] * X.shape[0])
y.to(device)
dl = DataLoader(TensorDataset(X, y), batch_size=3)
print("Pre-attack accuracy: ", Accuracy()(malconv, dl))
adv_dl = pdos_attack(malconv, dl)
print("Accuracy: ", Accuracy()(malconv, adv_dl))
