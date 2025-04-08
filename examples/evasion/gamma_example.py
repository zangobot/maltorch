from pathlib import Path

from secmlt.metrics.classification import Accuracy
from torch.utils.data import TensorDataset, DataLoader

from maltorch.adv.evasion.gamma_section_injection import GAMMASectionInjection
from maltorch.data.loader import load_from_folder, create_labels
from maltorch.zoo.malconv import MalConv

device = "cpu"

folder = Path(__file__).parent
X = load_from_folder(folder, "file")
X = X.to(device)
y = create_labels(X, 1)
y = y.to(device)
dl = DataLoader(TensorDataset(X, y), batch_size=3)

attack = GAMMASectionInjection(
    query_budget=20,
    benignware_folder=folder / "benignware",
    which_sections=[".text"],
    how_many_sections=3
)
model = MalConv.create_model(device=device)

print("Pre-attack accuracy: ", Accuracy()(model, dl))
adv_dl = attack(model, dl)
print("Accuracy: ", Accuracy()(model, adv_dl))
