from pathlib import Path

from secmlt.metrics.classification import Accuracy
from torch.utils.data import TensorDataset, DataLoader

from maltorch.adv.evasion.gamma_section_injection import GAMMASectionInjection
from maltorch.data.loader import load_from_folder, create_labels
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.malconv import MalConv

device = "mps"

exe_folder = Path(__file__).parent / ".." / "data" / "malware"
X = load_from_folder(exe_folder, "exe", device='cpu')
y = create_labels(X, 1, device='cpu')
dl = DataLoader(TensorDataset(X, y), batch_size=3)

attack = GAMMASectionInjection(
    query_budget=100,
    benignware_folder=exe_folder / ".." / "benignware",
    which_sections=[".text"],
    how_many_sections=10,
    device=device
)
model = MalConv.create_model(kernel_size=500, device=device)

print("Pre-attack accuracy: ", Accuracy()(model, dl))
adv_dl = attack(model, dl)
print("Accuracy: ", Accuracy()(model, adv_dl))
