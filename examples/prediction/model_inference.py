from pathlib import Path

from secmlt.metrics.classification import Accuracy
from torch.utils.data import DataLoader, TensorDataset

from maltorch.data.loader import load_from_folder, create_labels
from maltorch.data_processing.grayscale_preprocessing import GrayscalePreprocessing
from maltorch.data_processing.sigmoid_postprocessor import SigmoidPostprocessor
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.ember_gbdt import EmberGBDT
from maltorch.zoo.malconv import MalConv
from maltorch.zoo.resnet18 import ResNet18

# Specify the device to use (cpu, mps, cuda)
device = "cpu"

# Insert here the path to a file to evaluate
# exe_filepath = Path("path/to/exe/file/")
exe_folder = Path("path/to/exe/folder")

# For deep neural networks, we need to create a post-processor that
# generates probabilities from the logits output
sigmoid_postprocessor = SigmoidPostprocessor()

# Create the deep neural networks we want to evaluate.
# All the parameters of the networks are fetched online, since we are not passing
# the model_path into the create_model function.
networks = {
    'BBDnn': BBDnn.create_model(postprocessing=sigmoid_postprocessor, device=device),
    'Malconv': MalConv.create_model(postprocessing=sigmoid_postprocessor, device=device),
    'AvastStyleConv': AvastStyleConv.create_model(postprocessing=sigmoid_postprocessor, device=device),
    'EMBER GBDT': EmberGBDT.create_model(),
    'Grayscale ResNet18': ResNet18.create_model(
        preprocessing=GrayscalePreprocessing(),
        postprocessing=sigmoid_postprocessor,
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
