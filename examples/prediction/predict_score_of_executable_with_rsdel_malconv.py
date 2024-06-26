import torch
from secmlware.zoo.malconv import MalConv
from secmlware.data.loader import load_single_exe
from secmlware.data_processing.rsdel_preprocessing import RandomizedDeletionPreprocessing
from secmlware.data_processing.smoothing_postprocessing import SmoothingPostprocessing

exe_filepath = "path/to/exe/file/"
model_path = "/path/to/mode/state/dict"

preprocessing = RandomizedDeletionPreprocessing(
    pdel=0.97,
    num_versions=20,
    padding_value=256
)
postprocessing = SmoothingPostprocessing()
malconv = MalConv.create_model(
    model_path=model_path,
    preprocessing=preprocessing,
    postprocessing=postprocessing
)
x = load_single_exe(exe_filepath).to(torch.long).unsqueeze(0)
print(malconv(x).item())
