from pathlib import Path

from secmlt.metrics.classification import Accuracy
from secmlt.trackers import LossTracker, TensorboardTracker
from torch.utils.data import TensorDataset, DataLoader

from secmlware.adv.evasion.partialdos import PartialDOS
from secmlware.data.loader import load_from_folder, create_labels
from secmlware.zoo.malconv import MalConv

folder = Path(__file__).parent
X = load_from_folder(folder, "file", 1)
path = str(Path(__file__).parent / "logs" / "gf-pdos")
tensorboard_tracker = TensorboardTracker(path, [LossTracker()])
pdos_attack = PartialDOS(
    query_budget=30,
    trackers=tensorboard_tracker,
    random_init=False,
    backend="nevergrad",
    population_size=5,
)
malconv = MalConv.create_model()
y = create_labels(X, 1)
dl = DataLoader(TensorDataset(X, y), batch_size=3)
print("Pre-attack accuracy: ", Accuracy()(malconv, dl))
adv_dl = pdos_attack(malconv, dl)
print("Accuracy: ", Accuracy()(malconv, adv_dl))
