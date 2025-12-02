from pathlib import Path

from secmlt.metrics.classification import Accuracy
from torch.utils.data import DataLoader, TensorDataset

from maltorch.data.loader import load_from_folder, create_labels
from maltorch.data_processing.grayscale_preprocessing import GrayscalePreprocessing
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.ember_gbdt import EmberGBDT
from maltorch.zoo.malconv import MalConv
from maltorch.zoo.original_malconv import OriginalMalConv
from maltorch.zoo.resnet18 import ResNet18
from maltorch.zoo.thrember_gbdt import ThremberGBDT

# Specify the device to use (cpu, mps, cuda)
device = "cpu"

# Insert into this folder the malware to use for the evaluation
exe_folder = Path(__file__).parent / ".." / "data" / "malware"
models_folder = Path(__file__).parent / ".." / "data" / "models"

# Create the deep neural networks we want to evaluate.
# All the parameters of the networks are fetched online, since we are not passing
# the model_path into the create_model function.
networks = {
    'EMBER GBDT': EmberGBDT.create_model(),
    'THREMBER GBDT': ThremberGBDT.create_model(),
    'BBDnn': BBDnn.create_model(device=device),
    'Malconv': MalConv.create_model(device=device),
    'OriginalMalconv': OriginalMalConv.create_model(device=device),
    'AvastStyleConv': AvastStyleConv.create_model(device=device),
    'Grayscale ResNet18': ResNet18.create_model(
        preprocessing=GrayscalePreprocessing(),
        device=device),
}

# Load all the executables from the specified folder.
X = load_from_folder(exe_folder, "exe", device=device)
y = create_labels(X, 1)
data_loader = DataLoader(TensorDataset(X, y), batch_size=3)

# Compute all predictions, depending on the model
print("Computing maliciousness of loaded data...")
for k in networks:
    model = networks[k]
    print(f"{k}: {Accuracy()(model, data_loader) * 100:.2f}% malicious")
