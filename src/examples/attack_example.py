import torch
from torch.utils.data import TensorDataset, DataLoader

from src.adv.evasion.content_shift import ContentShift
from src.adv.evasion.partialdos import PartialDOS
from src.zoo.malconv import MalConv

device = "cpu"

pdos_attack = ContentShift(512, 10, 20)
malconv = MalConv.create_model(device=device)
x = torch.randint(0, 255, size=(10, 2**20))
x.to(device)
y = torch.LongTensor([1] * x.shape[0])
y.to(device)
dl = DataLoader(TensorDataset(x, y), batch_size=3)
adv_dl = pdos_attack(malconv, dl)
print("Done")
