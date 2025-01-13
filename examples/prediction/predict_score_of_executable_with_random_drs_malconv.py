import torch
from maltorch.zoo.malconv import MalConv
from maltorch.data.loader import load_single_exe
from maltorch.data_processing.random_drs_preprocessing import RandomDeRandomizedPreprocessing
from maltorch.data_processing.smoothing_postprocessing import SmoothingPostprocessing

exe_filepath = "path/to/exe/file/"
model_path = "/path/to/model/state/dict"

preprocessing = RandomDeRandomizedPreprocessing(
    file_percentage=0.05,
    num_chunks=100,
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
