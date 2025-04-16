from pathlib import Path

from secmlt.metrics.classification import Accuracy
from torch.utils.data import TensorDataset, DataLoader

from maltorch.adv.evasion.content_shift import ContentShift
from maltorch.data.loader import load_from_folder, create_labels
from maltorch.zoo.malconv import MalConv

folder = Path(__file__).parent
X = load_from_folder(folder, "exe")
y = create_labels(X, 1)
dl = DataLoader(TensorDataset(X, y), batch_size=2)

attack = ContentShift(
    query_budget=5, trackers=None, random_init=False
)
model = MalConv.create_model()

print("Pre-attack accuracy: ", Accuracy()(model, dl))
adv_dl = attack(model, dl)
print("Accuracy: ", Accuracy()(model, adv_dl))
