from pathlib import Path

from secmlt.metrics.classification import Accuracy
from secmlt.models.data_processing.identity_data_processing import IdentityDataProcessing
from torch.utils.data import DataLoader, TensorDataset

from maltorch.data.loader import load_from_folder, create_labels
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.original_malconv import OriginalMalConv

device = 'cpu'

exe_folder = Path(__file__).parent / ".." / "data" / "malware"
X = load_from_folder(exe_folder, device=device, limit=40)
print(X.shape)
y = create_labels(X, 1, device=device)
dl = DataLoader(TensorDataset(X, y), batch_size=1)

networks = {
    'OriginalMalconv': OriginalMalConv.create_model(device=device),
    'NoPaddingOriginalMalconv': OriginalMalConv.create_model(device=device, preprocessing=IdentityDataProcessing()),
    'BBDnn': BBDnn.create_model(device=device),
    'NoPaddingBBDnn': BBDnn.create_model(device=device, preprocessing=IdentityDataProcessing()),
}

for k in networks:
    print(k)
    model = networks[k]
    print("Accuracy:", Accuracy()(model, dl))
