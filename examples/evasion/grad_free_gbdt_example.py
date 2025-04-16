from pathlib import Path

from secmlt.metrics.classification import Accuracy
from secmlt.trackers import LossTracker, TensorboardTracker
from torch.utils.data import TensorDataset, DataLoader

from maltorch.adv.evasion.partialdos import PartialDOS
from maltorch.data.loader import load_from_folder, create_labels
from maltorch.zoo.ember_gbdt import EmberGBDT



folder = Path(__file__).parent
X = load_from_folder(folder, "file", limit=1)
path = str(Path(__file__).parent / "logs" / "gbdt-gf-pdos")
tensorboard_tracker = TensorboardTracker(path, [LossTracker()])
pdos_attack = PartialDOS(
    query_budget=30,
    trackers=tensorboard_tracker,
    random_init=False,
    backend="nevergrad",
    population_size=5,
)
gbdt = EmberGBDT.create_model()
y = create_labels(X, 1)
dl = DataLoader(TensorDataset(X, y), batch_size=3)
print("Pre-attack accuracy: ", Accuracy()(gbdt, dl))
adv_dl = pdos_attack(gbdt, dl)
print("Accuracy: ", Accuracy()(gbdt, adv_dl))
