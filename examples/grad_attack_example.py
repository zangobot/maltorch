from pathlib import Path

from secmlt.metrics.classification import Accuracy
from secmlt.trackers import TensorboardTracker, LossTracker
from torch.utils.data import TensorDataset, DataLoader

from secmlware.adv.evasion.partialdos import PartialDOS
from secmlware.data.loader import load_from_folder, create_labels
from secmlware.zoo.malconv import MalConv

device = "cpu"

folder = Path(__file__).parent
X = load_from_folder(folder, "file")
X = X.to(device)
y = create_labels(X, 1)
y = y.to(device)
dl = DataLoader(TensorDataset(X, y), batch_size=3)
path = str(Path(__file__).parent / "logs" / "pdos")
tensorboard_tracker = TensorboardTracker(path, [LossTracker()])
pdos_attack = PartialDOS(
    query_budget=20, trackers=tensorboard_tracker, random_init=False
)
malconv = MalConv.create_model(device=device)
print("Pre-attack accuracy: ", Accuracy()(malconv, dl))
adv_dl = pdos_attack(malconv, dl)
print("Accuracy: ", Accuracy()(malconv, adv_dl))
