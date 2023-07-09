import copy

import torch
from torch.nn import CrossEntropyLoss

from src.manipulations.replacement import ReplacementManipulation
from src.optim.bgd import BGD
from src.zoo.malconv import MalConv

malconv = MalConv.create_model(input_embedding=False)
x = torch.randint(0, 255, size=(5, 2**20))
xadv = copy.deepcopy(x)
manipulation_size = (5, 50)
delta = torch.randint(0, 255, manipulation_size)
indexes = torch.LongTensor(list(range(50)))
adv_x = ReplacementManipulation(torch.Tensor(indexes))(xadv, delta)
optimizer = BGD(
    [delta],
    malconv,
    lr=5,
    indexes_to_perturb=indexes
)
loss = CrossEntropyLoss()
labels = torch.LongTensor([0] * x.shape[0])
out = loss(malconv(adv_x), labels)
out.backward()
optimizer.step()
