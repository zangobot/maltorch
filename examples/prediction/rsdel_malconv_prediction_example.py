import torch
from maltorch.zoo.malconv import MalConv
from maltorch.data.loader import load_single_exe
from maltorch.data_processing.rsdel_preprocessing import RandomizedDeletionPreprocessing
from maltorch.data_processing.majority_voting_postprocessing import MajorityVotingPostprocessing

exe_filepath = "path/to/exe/file/"
model_path = "/path/to/mode/state/dict"

preprocessing = RandomizedDeletionPreprocessing(
    pdel=0.97,
    num_versions=100,
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
