import random
import torch
from maltorch.datasets.binary_dataset import BinaryDataset
import math


class KPartitionDeRandomizedSmoothingDataset(BinaryDataset):
    """
    Shoumik Saha, Weinxio Wang, Yigitcan Kaya, Soheil Feizi, Tudo Dumitras
    DRSM: De-Randomized Smoothing on Malware Classifier Providing Certified Robustness
    ICRL'24
    """
    def __init__(self,
             csv_filepath: str = None,
             goodware_directory: str = None,
             malware_directory: str = None,
             max_len: int = 2 ** 20,
             padding_idx: int = 256,
             min_len: int = None,
             num_chunks: int = 4,
             is_training: bool = True,
             sort_by_size: bool = False,
             min_chunk_size: int = 500
        ):
        super().__init__(
            csv_filepath=csv_filepath,
            goodware_directory=goodware_directory,
            malware_directory=malware_directory,
            max_len=max_len,
            min_len=min_len,
            padding_idx=padding_idx,
        )
        self.num_chunks = num_chunks
        self.min_chunk_size = min_chunk_size
        self.is_training = is_training

        if sort_by_size: #  Reorder files by size
            #sorted(self.all_files, key=lambda x: x[2])
            self.all_files.sort(key=lambda filename: filename[2])

    def pad_collate_func(self, batch):
        """
        This function splits a tensor of bytes into chunks.
        It works differently at training and at test time.
        During training, given an input example we randomly select a chunk of bytes
        During testing, given an input example we split the example into chunks.
        """
        vecs = []
        labels = []
        if self.is_training is True:  # Select a random chunk from a given executable
            for x, y in batch:
                L = x.shape[0]
                chunk_size = math.ceil(L/self.num_chunks)
                if chunk_size < self.min_chunk_size:
                    chunk_size = self.min_chunk_size
                start_location = random.randint(0, max(0, x.shape[0] - chunk_size))
                end_location = start_location + chunk_size
                vecs.append(x[start_location: end_location])
                labels.append(y)
            x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True, padding_value=self.padding_idx)
            # stack will give us (B, 1), so index [:,0] to get to just (B)
            y = torch.tensor(labels)
            return x, y
        else:  # Split an executable into chunks
            if len(batch) == 1: # Only implemented for batch sizes equals to 1
                x = batch[0][0]
                L = len(x)
                chunk_size = math.ceil(L/self.num_chunks)
                if chunk_size < self.min_chunk_size:
                    chunk_size = self.min_chunk_size
                x = [x[i:i+chunk_size] for i in range(0, L, chunk_size)]
                x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.padding_idx)
                y = batch[0][1]
                return x, y
            else:
                raise NotImplementedError
