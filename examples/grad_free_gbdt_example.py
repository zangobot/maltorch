from pathlib import Path

from secmlt.metrics.classification import Accuracy
# from secmlt.trackers import LossTracker, TensorboardTracker
from torch.utils.data import TensorDataset, DataLoader

from secmlware.adv.evasion.partialdos import PartialDOS
from secmlware.data.loader import load_from_folder, create_labels, load_single_exe
from secmlware.zoo.ember_gbdt import EmberGBDT

tree_path = (
    Path(__file__).parent.parent
    / "src"
    / "secmlware"
    / "zoo"
    / "models"
    / "ember_model.txt"
)

# folder = Path(__file__).parent
folder = Path("/Users/bridge/PhD/Code/mal-pipeline/malware/da5d8fb5aa21cd6ffcacd9bc6cfc4305ab96ed20051273a68137785d838230f7")
X = load_single_exe(folder).unsqueeze(0)
# X = load_from_folder(folder, "file", limit=3)
path = str(Path(__file__).parent / "logs" / "gbdt-gf-pdos")
# tensorboard_tracker = TensorboardTracker(path, [LossTracker()])
pdos_attack = PartialDOS(
    query_budget=30,
    trackers=None,
    random_init=False,
    backend="nevergrad",
    population_size=5,
)
gbdt = EmberGBDT.create_model(str(tree_path))
y = create_labels(X, 1)
dl = DataLoader(TensorDataset(X, y), batch_size=3)
print("Pre-attack accuracy: ", Accuracy()(gbdt, dl))
adv_dl = pdos_attack(gbdt, dl)
print("Accuracy: ", Accuracy()(gbdt, adv_dl))
