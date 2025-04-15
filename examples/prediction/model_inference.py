from pathlib import Path

import torch

from maltorch.data.loader import load_single_exe
from maltorch.data_processing.grayscale_preprocessing import GrayscalePreprocessing
from maltorch.data_processing.sigmoid_postprocessor import SigmoidPostprocessor
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.ember_gbdt import EmberGBDT
from maltorch.zoo.malconv import MalConv
from maltorch.zoo.resnet18 import ResNet18

# Insert here the path to a file to evaluate
exe_filepath = Path("../evasion/petya.file")
# exe_filepath = Path("path/to/exe/file/")

# For deep neural networks, we need to create a post-processor that
# generates probabilities from the logits output
sigmoid_postprocessor = SigmoidPostprocessor()

# Create the deep neural networks we want to evaluate.
# All the parameters of the networks are fetched online, since we are not passing
# the model_path into the create_model function.
networks = {
    'BBDnn': BBDnn.create_model(postprocessing=sigmoid_postprocessor),
    'Malconv': MalConv.create_model(postprocessing=sigmoid_postprocessor),
    'AvastStyleConv': AvastStyleConv.create_model(postprocessing=sigmoid_postprocessor),
    'EMBER GBDT': EmberGBDT.create_model(),
    'Grayscale ResNet18': ResNet18.create_model(
        preprocessing=GrayscalePreprocessing(),
        postprocessing=sigmoid_postprocessor)
}

# Load a single sample, and put into batch to allow inference
x = load_single_exe(exe_filepath).to(torch.long).unsqueeze(0)

# Compute all predictions, depending on the model
print(f"Computing maliciousness of {exe_filepath}")
for k in networks:
    model = networks[k]
    print(f"{k}: {model(x).item() * 100:.4f}% malicious")
