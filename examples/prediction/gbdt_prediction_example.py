import torch
from maltorch.zoo.ember_gbdt import EmberGBDT
from maltorch.data.loader import load_single_exe


exe_filepath = "path/to/exe/file/"
# model_path = "/path/to/model/state/dict"
model_path = None # will download the model from Google Drive
classifier = EmberGBDT.create_model(
    model_path=model_path,
)
x = load_single_exe(exe_filepath).to(torch.long).unsqueeze(0)
print(classifier.predict(x).item())

