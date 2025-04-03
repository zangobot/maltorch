from pathlib import Path

from secmlt.metrics.classification import Accuracy
from torch.utils.data import TensorDataset, DataLoader

from maltorch.adv.evasion.content_shift import ContentShift
from maltorch.adv.evasion.partialdos import PartialDOS
from maltorch.data.loader import load_from_folder, create_labels
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.malconv import MalConv

device = "cpu"

folder = Path(__file__).parent
X = load_from_folder(folder, "file")
X = X.to(device)
y = create_labels(X, 1)
y = y.to(device)
dl = DataLoader(TensorDataset(X, y), batch_size=3)

pdos_attack = ContentShift(
    query_budget=20, trackers=None, random_init=False
)
model = MalConv.create_model(device=device)

print("Pre-attack accuracy: ", Accuracy()(model, dl))
adv_dl = pdos_attack(model, dl)
print("Accuracy: ", Accuracy()(model, adv_dl))
