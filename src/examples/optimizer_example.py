import copy

import torch
from torch.nn import CrossEntropyLoss

from src.optim.bgd import BGD
from src.zoo.malconv import MalConv

malconv = MalConv.create_model(input_embedding=True)
x = torch.randint(0, 255, size=(5, 2**20))
xadv = copy.deepcopy(x)
emb_x: torch.Tensor = malconv.embed(x)
delta = torch.zeros_like(emb_x, requires_grad=True)
delta.retain_grad()
adv_x = emb_x + delta
optimizer = BGD([delta], lr=5, embedding_matrix=malconv.embedding_matrix())
loss = CrossEntropyLoss()
labels = torch.LongTensor([0] * x.shape[0])
out = loss(malconv(adv_x), labels)
out.backward()
optimizer.step()
# grad_emb_x = torch.autograd.grad(out, emb_x, retain_graph=True, only_inputs=False)
print(delta.grad)
