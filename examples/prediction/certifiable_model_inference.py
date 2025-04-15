from pathlib import Path

import torch

from maltorch.data.loader import load_single_exe
from maltorch.data_processing.majority_voting_postprocessing import MajorityVotingPostprocessing
from maltorch.data_processing.rsdel_preprocessing import RandomizedDeletionPreprocessing
from maltorch.utils.utils import download_gdrive
from maltorch.zoo.malconv import MalConv

# Insert here the path to a file to evaluate
exe_filepath = Path("../evasion/petya.file")
# exe_filepath = Path("path/to/exe/file/")

# Create pre- and post-processors for certifiable approaches.
rsdel_preprocessing = RandomizedDeletionPreprocessing(pdel=0.03, num_versions=100)
majority_voting = MajorityVotingPostprocessing(apply_sigmoid=True)

rsdel_malconv_path = str(Path(__file__).parent / "rsdel_malconv")
download_gdrive(gdrive_id="1Ste3BBC5eONw42-tih2zjhm4Rj0ck-P6", fname_save=rsdel_malconv_path)

# Create the deep neural networks we want to evaluate.
# All the parameters of the networks are fetched online, since we are not passing
# the model_path into the create_model function.
networks = {
    'MalConv RSDel': MalConv.create_model(model_path=rsdel_malconv_path,
                                          max_len=2000000,
                                          kernel_size=512,
                                          stride=512,
                                          preprocessing=rsdel_preprocessing,
                                          postprocessing=majority_voting)
}

# Load a single sample, and put into batch to allow inference
x = load_single_exe(exe_filepath).to(torch.long).unsqueeze(0)

# Compute all predictions, depending on the model
print(f"Computing maliciousness of {exe_filepath}")
for k in networks:
    model = networks[k]
    print(f"{k}: {model(x).item() * 100:.4f}% malicious")