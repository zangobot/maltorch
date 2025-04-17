from pathlib import Path

from secmlt.metrics.classification import Accuracy
from torch.utils.data import TensorDataset, DataLoader

from maltorch.adv.evasion.partialdos import PartialDOS
from maltorch.data.loader import load_from_folder, create_labels
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.malconv import MalConv

device = "cpu"

exe_folder = Path("path/to/exe/folder")
X = load_from_folder(exe_folder, "exe",device=device)
y = create_labels(X, 1, device=device)
dl = DataLoader(TensorDataset(X, y), batch_size=1)

grad_attack = PartialDOS(
    query_budget=20, trackers=None, random_init=False
)
# Create the deep neural networks we want to evaluate.
# All the parameters of the networks are fetched online, since we are not passing
# the model_path into the create_model function.
# Also, differently from inference examples, we do not need to use the SigmoidPostprocessor
# as it is already included inside the loss of the attack.
networks = {
    'BBDnn': BBDnn.create_model(),
    'Malconv': MalConv.create_model(),
    'AvastStyleConv': AvastStyleConv.create_model(),
}
for k in networks:
    print(k)
    model = networks[k]
    print("- - - Pre-attack accuracy: ", Accuracy()(model, dl))
    adv_dl = grad_attack(model, dl)
    print("- - - Accuracy: ", Accuracy()(model, adv_dl))
