from pathlib import Path

from secmlt.metrics.classification import Accuracy
from torch.utils.data import TensorDataset, DataLoader

from maltorch.adv.evasion.base_optim_attack_creator import OptimizerBackends
from maltorch.adv.evasion.content_shift import ContentShift
from maltorch.data.loader import load_from_folder, create_labels
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.ember_gbdt import EmberGBDT
from maltorch.zoo.malconv import MalConv

device = "cpu"

exe_folder = Path(__file__).parent / ".." / "data" / "malware"
X = load_from_folder(exe_folder, "exe", device='cpu')
y = create_labels(X, 1, device='cpu')
dl = DataLoader(TensorDataset(X, y), batch_size=4)

# We instantiate the nevergrad version of the attack.
# Not all the attacks provide both implementations.
query_budget = 500
pdos_attack = ContentShift(
    query_budget=query_budget,
    perturbation_size=4096,
    trackers=None,
    random_init=False,
    backend=OptimizerBackends.NG,
    population_size=10,
    device=device
)
# Create the deep neural networks we want to evaluate.
# All the parameters of the networks are fetched online, since we are not passing
# the model_path into the create_model function.

networks = {
    'BBDnn': BBDnn.create_model(device=device),
    'Malconv': MalConv.create_model(device=device),
    'AvastStyleConv': AvastStyleConv.create_model(),
}
for k in networks:
    print(k)
    model = networks[k]
    print("- - - Pre-attack accuracy: ", Accuracy()(model, dl))
    adv_dl = pdos_attack(model, dl)
    print("- - - Accuracy: ", Accuracy()(model, adv_dl))

#The GBDT already outputs probabilities, so we need to use the Accuracy object
# to calculate the real accuracy.
# Also, for the same reason, the loss should take into account this change.
# Hence, we set the loss of the attack to be the BCELoss


other = {
    'EMBER GBDT': EmberGBDT.create_model()
}
pdos_attack = ContentShift(
    query_budget=query_budget,
    perturbation_size=4096,
    trackers=None,
    random_init=False,
    backend=OptimizerBackends.NG,
    population_size=3,
    model_outputs_logits=False,
    device='cpu'
)
for k in other:
    print(k)
    model = other[k]
    print("- - - Pre-attack accuracy: ", Accuracy()(model, dl))
    adv_dl = pdos_attack(model, dl)
    print("- - - Accuracy: ", Accuracy()(model, adv_dl))