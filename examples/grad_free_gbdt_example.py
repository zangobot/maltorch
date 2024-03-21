from pathlib import Path

from secmlt.metrics.classification import Accuracy
from secmlt.trackers import LossTracker, TensorboardTracker, ScoresTracker
from torch.utils.data import TensorDataset, DataLoader

from secmlware.adv.evasion.partialdos import PartialDOSGradFree
from secmlware.data.loader import load_from_folder, create_labels
from secmlware.zoo.ember_gbdt import EmberGBDT

tree_path = Path(__file__).parent.parent / "src" / "secmlware" / "zoo" / "models" / "ember_model.txt"

folder = Path(__file__).parent
X = load_from_folder(folder, "file")
path = str(Path(__file__).parent / "logs" / "gbdt-gf-pdos")
tensorboard_tracker = TensorboardTracker(path, [LossTracker()])
pdos_attack = PartialDOSGradFree(query_budget=50)
gbdt = EmberGBDT.create_model(str(tree_path))
y = create_labels(X, 1)
dl = DataLoader(TensorDataset(X, y), batch_size=3)
print("Pre-attack accuracy: ", Accuracy()(gbdt, dl))
adv_dl = pdos_attack(gbdt, dl)
print("Accuracy: ", Accuracy()(gbdt, adv_dl))
