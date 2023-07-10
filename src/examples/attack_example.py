import torch
from torch.utils.data import TensorDataset, DataLoader

from src.adv.evasion.partialdos import PartialDOS
from src.zoo.malconv import MalConv

device = "cpu"

pdos_attack = PartialDOS(10, 20)
malconv = MalConv.create_model(device=device)
x = torch.randint(0, 255, size=(5, 2**20))
x.to(device)
print(malconv(x))
y = torch.LongTensor([1, 1, 1, 1, 1])
y.to(device)
dl = DataLoader(TensorDataset(x, y))
adv_dl = pdos_attack(malconv, dl)
print("Done")
