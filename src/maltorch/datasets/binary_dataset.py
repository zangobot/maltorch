from torch.utils.data import Dataset
from typing import Tuple
import torch
import os
from pathlib import Path


class BinaryDataset(Dataset):
    def __init__(self,
                 csv_filepath: str = None,
                 goodware_directory: str = None,
                 malware_directory: str = None,
                 max_len: int = None,
                 padding_idx: int = 256,
                 min_len: int = None):
        self.all_files = []
        self.max_len = max_len
        self.min_len = min_len
        self.padding_idx = padding_idx

        if csv_filepath is not None:
            with open(csv_filepath, "r") as input_file:
                lines = input_file.readlines()
                for line in lines:
                    tokens = line.strip().split(",")
                    self.all_files.append([tokens[0], int(tokens[1]), os.path.getsize(tokens[0])])
        elif goodware_directory is not None and malware_directory is not None:
            self.all_files.extend(
                [[os.path.join(goodware_directory, filename), 0, os.path.getsize(os.path.join(goodware_directory, filename))] for filename in os.listdir(goodware_directory)])
            self.all_files.extend(
                [[os.path.join(malware_directory, filename), 1, os.path.getsize(os.path.join(malware_directory, filename))] for filename in os.listdir(malware_directory)])
        else:
            raise NotImplementedError("You need to either provide CSV file containing (sample,label) "
                                      "or the paths where the goodware and malware are stored.")

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        to_load, label, _ = self.all_files[index]
        x = load_single_exe(to_load, max_len=self.max_len, min_len=self.min_len, padding_idx=self.padding_idx)
        return x, torch.tensor(label)

    def pad_collate_func(self, batch):
        """
        This should be used as the collate_fn=pad_collate_func for a pytorch DataLoader object in order to pad out files in a batch to the length of the longest item in the batch.
        """
        vecs = [x[0] for x in batch]
        labels = [x[1] for x in batch]

        x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True, padding_value=self.padding_idx)
        # stack will give us (B, 1), so index [:,0] to get to just (B)
        y = torch.stack(labels)

        return x, y


def load_single_exe(path: Path, max_len: int = None, min_len: int = None, padding_idx: int = 256) -> torch.Tensor:
    """
    Create a torch.Tensor from the file pointed in the path
    :param path: a pathlib Path
    :return: torch.Tensor containing the bytes of the file as a tensor
    """
    with open(path, "rb") as h:
        if max_len is None:
            code = h.read()
        else:
            code = h.read(max_len)
    x = torch.frombuffer(bytearray(code), dtype=torch.uint8)
    x = x.to(torch.long)
    if min_len is not None: # Pad the tensor to the minimum length - required for some architectures
        if x.shape[0] < min_len:
            padding_idx = torch.tensor(padding_idx, dtype=torch.long)
            x = torch.nn.functional.pad(x, (0, min_len - x.shape[0]), mode='constant', value=padding_idx)
    return x
