from pathlib import Path

from secmlt.metrics.classification import Accuracy
from torch.utils.data import DataLoader, TensorDataset

from maltorch.data.loader import load_from_folder, create_labels
from maltorch.data_processing.majority_voting_postprocessing import MajorityVotingPostprocessing
from maltorch.data_processing.rsdel_preprocessing import RandomizedDeletionPreprocessing
from maltorch.data_processing.rs_preprocessing import RandomizedAblationPreprocessing
from maltorch.utils.utils import download_gdrive
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.malconv import MalConv

# Specify the device to use (cpu, mps, cuda)
device = "cpu"

# Insert into this folder the malware to use for the evaluation
exe_folder = Path(__file__).parent / ".." / "data" / "malware"

# Create pre- and post-processors for certifiable approaches.
rsdel_preprocessing = RandomizedDeletionPreprocessing(pdel=0.03, num_versions=100)
rs_preprocessing = RandomizedAblationPreprocessing(pabl=0.2, num_versions=100)
majority_voting = MajorityVotingPostprocessing(apply_sigmoid=True)

rsdel_malconv_path = Path(__file__).parent / "rsdel_malconv"
rs_malconv_path = Path(__file__).parent / "rs_malconv"
rs_bbdnn_path = Path(__file__).parent / "rs_bbdnn"
rsdel_bbdnn_path = Path(__file__).parent / "rsdel_bbdnn"

if not rsdel_malconv_path.exists():
    download_gdrive(gdrive_id="1Ste3BBC5eONw42-tih2zjhm4Rj0ck-P6", fname_save=rsdel_malconv_path)
if not rs_malconv_path.exists():
    download_gdrive(gdrive_id="1xi-Dc758WxpuNNkfVJdpAGqEzd9Dx-rS", fname_save=rs_malconv_path)
if not rs_bbdnn_path.exists():
    download_gdrive(gdrive_id="1lpsnFz7hFSrLXrguxRuu5Mrmarp9uNod", fname_save=rs_bbdnn_path)
if not rsdel_bbdnn_path.exists():
    download_gdrive(gdrive_id="1Kq7HngMv4cVD0HjpaqHctqEpQlicOjW_", fname_save=rsdel_bbdnn_path)

# Create the deep neural networks we want to evaluate.
# All the parameters of the networks are fetched online, since we are not passing
# the model_path into the create_model function.
networks = {
    'MalConv RSDel': MalConv.create_model(model_path=str(rsdel_malconv_path),
                                          max_len=2000000,
                                          kernel_size=512,
                                          stride=512,
                                          preprocessing=rsdel_preprocessing,
                                          postprocessing=majority_voting,
                                          device=device),
    'MalConv RS': MalConv.create_model(model_path=str(rs_malconv_path),
                                       max_len=2000000,
                                       kernel_size=500,
                                       stride=500,
                                       preprocessing=rs_preprocessing,
                                       postprocessing=majority_voting,
                                       device=device),
    'BBDNN RS': BBDnn.create_model(model_path=str(rs_bbdnn_path),
                                   preprocessing=rs_preprocessing,
                                   postprocessing=majority_voting,
                                   device=device),
    'BBDNN RSDel': BBDnn.create_model(model_path=str(rsdel_bbdnn_path),
                                      preprocessing=rsdel_preprocessing,
                                      postprocessing=majority_voting,
                                      device=device)
}

# Load all the executables from the specified folder.
X = load_from_folder(exe_folder, "exe", device=device)
y = create_labels(X, 1, device=device)

# BEWARE that currently all the certification methods only works with batch size 1!
data_loader = DataLoader(TensorDataset(X, y), batch_size=1)

# Compute all predictions, depending on the model
print("Computing maliciousness of loaded data...")
for k in networks:
    model = networks[k]
    print(f"{k}: {Accuracy()(model, data_loader) * 100:.2f}% malicious")
