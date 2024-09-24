import torch
from secmlware.zoo.malconv import MalConv
from secmlware.data.loader import load_single_exe
from secmlware.data_processing.rs_preprocessing import RandomizedAblationPreprocessing
from secmlware.data_processing.smoothing_postprocessing import SmoothingPostprocessing

exe_filepath = "path/to/exe/file/"
model_path = "/path/to/model/state/dict"

preprocessing = RandomizedAblationPreprocessing(
    pabl=0.97,
    num_versions=100,
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
