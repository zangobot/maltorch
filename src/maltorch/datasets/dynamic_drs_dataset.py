import random
import torch
from maltorch.datasets.binary_dataset import BinaryDataset
from abc import ABC, abstractmethod
import torch.nn.functional as F


class DynamicChunkSizeDRSDataset(BinaryDataset, ABC):
    """
    Daniel Gibert, Giulio Zizzo, Quan Le, Jordi Planes
    Adversarial Robustness of Deep Learning-based Malware Detectors via (De) Randomized Smoothing
    IEEE Access 2024
    """
    def __init__(self,
                 csv_filepath: str = None,
                 goodware_directory: str = None,
                 malware_directory: str = None,
                 max_len: int = 2 ** 20,
                 padding_idx: int = 256,
                 min_len: int = None,
                 sort_by_size: bool = False,
                 file_percentage: float = 0.05,
                 num_chunks: int = 100,
                 is_training: bool = True,
                 min_chunk_size=500):
        super().__init__(
            csv_filepath=csv_filepath,
            goodware_directory=goodware_directory,
            malware_directory=malware_directory,
            max_len=max_len,
            padding_idx=padding_idx,
            min_len=min_len
        )
        self.is_training = is_training
        self.file_percentage = file_percentage
        self.num_chunks = num_chunks
        self.min_chunk_size = min_chunk_size
        self.sort_by_size = sort_by_size

        if sort_by_size: #  Reorder files by size
            #sorted(self.all_files, key=lambda x: x[2])
            self.all_files.sort(key=lambda filename: filename[2])

    def pad_collate_func(self, batch):
        """
        This function splits a tensor of bytes into chunks.
        It works differently at training and at test time.
        During training, given an input example we randomly select a chunk of bytes
        During testing, given an input example we sequentially extract N chunks of bytes. This chunks may overlap with
        one another depending on the amount of chunks you want to extract and the size of each chunk.
        """
        if self.is_training is True:  # Select a random chunk from a given executable
            return self.generate_training_examples(batch)
        else:  # Split an executable into chunks
            return self.generate_testing_examples(batch)

    def generate_training_examples(self, batch):
        vecs = []
        labels = []
        max_size = max([x.size()[0] for x, y in batch])
        chunk_size = int(-(-(max_size * self.file_percentage) // 1))

        for x, y in batch:
            # Code here
            if x.shape[0] <= chunk_size:
                vecs.append(x)
            else:
                start_location = random.randint(0, max(0, x.shape[0] - chunk_size))
                end_location = start_location + chunk_size
                vecs.append(x[start_location:end_location])
            labels.append(y)

        # Pad each chunk to `min_len` if necessary
        padded_Xs = []
        for x in vecs:
            if x.shape[0] < self.min_len:
                padding_size = self.min_len - x.shape[0]
                # Pad the vector with padding_value (or any other padding value you prefer)
                padded_x = F.pad(x, (0, padding_size), value=self.padding_idx)
                padded_Xs.append(padded_x)
            else:
                padded_Xs.append(x)

        x = torch.nn.utils.rnn.pad_sequence(padded_Xs, batch_first=True, padding_value=self.padding_idx)
        # stack will give us (B, 1), so index [:,0] to get to just (B)
        y = torch.tensor(labels)
        return x, y

    @abstractmethod
    def generate_testing_examples(self, batch):
        ...

