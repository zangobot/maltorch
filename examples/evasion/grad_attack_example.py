from pathlib import Path

import torch
from secmlt.metrics.classification import Accuracy
from secmlt.trackers import ScoresTracker
from torch.utils.data import TensorDataset, DataLoader

from maltorch.adv.evasion.partialdos import PartialDOS
from maltorch.data.loader import load_from_folder, create_labels
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.malconv import MalConv
from maltorch.zoo.original_malconv import OriginalMalConv

import lief

lief.logging.disable()

device = "cpu"

exe_folder = Path(__file__).parent / ".." / "data" / "malware"
X = load_from_folder(exe_folder, device=device)
y = create_labels(X, 1, device=device)
dl = DataLoader(TensorDataset(X, y), batch_size=32)

tracker = ScoresTracker()

query_budget = 10

grad_attack = PartialDOS(
    query_budget=query_budget,
    random_init=False,
    device=device,
)
# Create the deep neural networks we want to evaluate.
# All the parameters of the networks are fetched online, since we are not passing
# the model_path into the create_model function.
# Also, differently from inference examples, we do not need to use the SigmoidPostprocessor
# as it is already included inside the loss of the attack.

networks = {
    'OriginalMalconvSecMLMalware': OriginalMalConv.create_model(device=device),
    'Malconv': MalConv.create_model(device=device),
    'BBDnn': BBDnn.create_model(device=device),
    'AvastStyleConv': AvastStyleConv.create_model(device=device)
}
for k in networks:
    print(k)
    model = networks[k]
    print("- - - Pre-attack accuracy: ", Accuracy()(model, dl))
    adv_dl = grad_attack(model, dl)
    print("- - - Accuracy: ", Accuracy()(model, adv_dl))
