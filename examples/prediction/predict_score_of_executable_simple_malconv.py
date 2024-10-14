import torch
from secmlware.zoo.malconv import MalConv
from secmlware.data.loader import load_single_exe

exe_filepath = "path/to/exe/file/"
# model_path = "/path/to/model/state/dict"
model_path = None # will download the model from Google Drive

malconv = MalConv.create_model()
x = load_single_exe(exe_filepath).to(torch.long).unsqueeze(0)
print(malconv(x).item())
