from pathlib import Path

from secmlt.metrics.classification import Accuracy
from torch.utils.data import TensorDataset, DataLoader

from maltorch.adv.evasion.gamma_section_injection import GAMMASectionInjection
from maltorch.data.loader import load_from_folder, create_labels
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.malconv import MalConv
from maltorch.zoo.original_malconv import OriginalMalConv

device = "mps"

exe_folder = Path(__file__).parent / ".." / "data" / "malware"
X = load_from_folder(exe_folder, device=device)
y = create_labels(X, 1, device=device)
dl = DataLoader(TensorDataset(X, y), batch_size=3)

query_budget = 100
how_many_sections = 30

attack = GAMMASectionInjection(
    query_budget=query_budget,
    benignware_folder=exe_folder / ".." / "benignware",
    which_sections=[".rdata"],
    how_many_sections=how_many_sections,
    device=device
)

networks = {
    'AndersonMalconv': OriginalMalConv.create_model(device=device),
    'Malconv': MalConv.create_model(device=device),
    'BBDnn': BBDnn.create_model(device=device),
    'AvastStyleConv': AvastStyleConv.create_model(device=device)
}
for k in networks:
    print(k)
    model = networks[k]
    print("Pre-attack accuracy: ", Accuracy()(model, dl))
    adv_dl = attack(model, dl)
    print("Accuracy: ", Accuracy()(model, adv_dl))
