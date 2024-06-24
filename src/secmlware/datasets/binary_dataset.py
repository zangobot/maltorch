from torch.utils.data import Dataset
from random import shuffle
from typing import Tuple
import torch
import os


class BinaryDataset(Dataset):
    def __init__(self,
                 csv_filepath: str = None,
                 goodware_directory: str = None,
                 malware_directory: str = None,
                 max_len: int = 2**20,
                 padding_value: float = 256.0):
        self.all_files = []
        self.max_len = max_len
        self.padding_value = padding_value

        if csv_filepath is not None:
            with open(csv_filepath, "r") as input_file:
                lines = input_file.readlines()
                for line in lines:
                    tokens = line.strip().split(",")
                    self.all_files.append([tokens[0], int(tokens[1])])
        elif goodware_directory is not None and malware_directory is not None:
            self.all_files.extend(
                [[os.path.join(goodware_directory, filename), 0] for filename in os.listdir(goodware_directory)])
            self.all_files.extend(
                [[os.path.join(malware_directory, filename), 1] for filename in os.listdir(malware_directory)])
        else:
            raise NotImplementedError("You need to either provide CSV file containing (sample,id) "
                                      "or the paths where the goodware and malware are stored.")

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        to_load, label = self.all_files[index]
        x = load_single_exe(to_load, max_len=self.max_len)
        return x, torch.tensor(label)

    def pad_collate_func(self, batch):
        """
        This should be used as the collate_fn=pad_collate_func for a pytorch DataLoader object in order to pad out files in a batch to the length of the longest item in the batch.
        """
        vecs = [x[0] for x in batch]
        labels = [x[1] for x in batch]

        x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True, padding_value=self.padding_value)
        # stack will give us (B, 1), so index [:,0] to get to just (B)
        y = torch.stack(labels)

        return x, y



# This should go to secmlware.loader.py - Ask Luca
from pathlib import Path

def load_single_exe(path: Path, max_len: int = 2**20) -> torch.Tensor:
    """
    Create a torch.Tensor from the file pointed in the path
    :param path: a pathlib Path
    :return: torch.Tensor containing the bytes of the file as a tensor
    """
    with open(path, "rb") as h:
        code = h.read(max_len)
    x = torch.frombuffer(bytearray(code), dtype=torch.uint8).to(torch.long)
    return x
