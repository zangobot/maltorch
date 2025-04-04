import torch
from maltorch.zoo.malconv import MalConv
from maltorch.data.loader import load_single_exe
from maltorch.data_processing.drs_preprocessing import DeRandomizedPreprocessing
from maltorch.data_processing.majority_voting_postprocessing import MajorityVotingPostprocessing

exe_filepath = "path/to/exe/file/"
model_path = "/path/to/model/state/dict"

preprocessing = DeRandomizedPreprocessing(
    chunk_size=512,
    padding_idx=256
)
postprocessing = MajorityVotingPostprocessing()
model = MalConv.create_model(
    model_path=model_path,
    preprocessing=preprocessing,
    postprocessing=postprocessing
)
x = load_single_exe(exe_filepath).to(torch.long).unsqueeze(0)
print(model(x).item())
