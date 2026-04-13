import torch
from torch.nn.utils.rnn import pad_sequence

from maltorch.initializers.initializers import ByteBasedInitializer
from maltorch.utils.pe_operations import padding_manipulation


class PaddingInitializer(ByteBasedInitializer):
    def __init__(self, padding: int = 2048, random_init: bool = False):
        super().__init__(random_init)
        self.padding = padding

    def __call__(self, x: torch.Tensor):
        X = []
        deltas = []
        indexes = []

        for x_i in x:
            x_i, padding_indexes = padding_manipulation(x_i.unsqueeze(0), self.padding)

            # [1, D] -> [D]
            x_i = x_i.squeeze(0)
            X.append(x_i)

            delta = (
                torch.zeros(len(padding_indexes), dtype=torch.float32)
                if not self.random_init
                else torch.randint(0, 255, (len(padding_indexes),), dtype=torch.float32)
            )
            deltas.append(delta)

            indexes.append(torch.tensor(padding_indexes, dtype=torch.long))

        device = x.device

        # pad samples to same length
        x = pad_sequence(X, batch_first=True, padding_value=256)
        delta = pad_sequence(deltas, batch_first=True, padding_value=0)
        indexes = pad_sequence(indexes, batch_first=True, padding_value=-1)

        x = x.long().to(device)
        delta = delta.to(device)
        indexes = indexes.to(device)

        return x, delta, indexes