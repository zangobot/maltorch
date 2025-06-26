import torch
import lief
from maltorch.initializers.initializers import ByteBasedInitializer
from maltorch.utils.utils import convert_torch_exe_to_list


class PartialDOSInitializer(ByteBasedInitializer):
    def __init__(self, random_init: bool = False):
        super().__init__(random_init)

    def __call__(self, x: torch.Tensor):
        indexes = torch.arange(2, 60).long().repeat((x.shape[0], 1))
        delta = (
            torch.zeros((x.shape[0], 58))
            if not self.random_init
            else torch.randint(0, 255, (x.shape[0], 58))
        ).float()
        return x, delta, indexes


class DOSHeaderStubInitializer(ByteBasedInitializer):
    def __init__(self, random_init: bool = False):
        super().__init__(random_init)

    def __call__(self, x: torch.Tensor):
        indexes = []
        for i in range(x.shape[0]):
            lief_x =  lief.PE.parse(convert_torch_exe_to_list(x[i]))
            pe_position = lief_x.dos_header.addressof_new_exeheader
            x_indexes = list(range(2, 60)) + list(range(64, pe_position))
            indexes.append(torch.LongTensor(x_indexes))
        indexes = torch.nn.utils.rnn.pad_sequence(indexes, padding_value=-1).transpose(0, 1).long()
        delta = (
            torch.zeros_like(indexes)
            if not self.random_init
            else torch.randint(0, 255, indexes.shape)
        ).float()
        delta[indexes == -1] = -1
        delta = delta.float()
        delta = delta.to(x.device)
        indexes = indexes.to(x.device)
        return x, delta, indexes