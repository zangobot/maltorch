from pathlib import Path

from secmlt.metrics.classification import Accuracy
from torch.utils.data import TensorDataset, DataLoader

from maltorch.adv.evasion.base_optim_attack_creator import OptimizerBackends
from maltorch.adv.evasion.partialdos import PartialDOS
from maltorch.data.loader import load_from_folder, create_labels
from maltorch.zoo.malconv import MalConv

folder = Path(__file__).parent
X = load_from_folder(folder, "file")
pdos_attack = PartialDOS(
    query_budget=30,
    random_init=False,
    backend=OptimizerBackends.NG,
    population_size=5,
)
malconv = MalConv.create_model()
y = create_labels(X, 1)
dl = DataLoader(TensorDataset(X, y), batch_size=3)
print("Pre-attack accuracy: ", Accuracy()(malconv, dl))
adv_dl = pdos_attack(malconv, dl)
print("Accuracy: ", Accuracy()(malconv, adv_dl))
