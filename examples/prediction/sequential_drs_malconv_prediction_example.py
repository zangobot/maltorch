import torch
from maltorch.zoo.malconv import MalConv
from maltorch.data.loader import load_single_exe
from maltorch.data_processing.sequential_drs_preprocessing import SequentialDeRandomizedPreprocessing
from maltorch.data_processing.majority_voting_postprocessing import MajorityVotingPostprocessing

exe_filepath = "path/to/exe/file/"
model_path = "/path/to/model/state/dict"

preprocessing = SequentialDeRandomizedPreprocessing(
    file_percentage=0.05,
    num_chunks=100,
    padding_idx=256
)
postprocessing = MajorityVotingPostprocessing()
malconv = MalConv.create_model(
    model_path=model_path,
    preprocessing=preprocessing,
    postprocessing=postprocessing
)
x = load_single_exe(exe_filepath).to(torch.long).unsqueeze(0)
print(malconv(x).item())
