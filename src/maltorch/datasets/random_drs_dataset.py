from typing import Tuple

import torch
from random import randint
from maltorch.datasets.dynamic_drs_dataset import DynamicChunkSizeDRSDataset

class RandomDRSDataset(DynamicChunkSizeDRSDataset):
    """
    Daniel Gibert, Giulio Zizzo, Quan Le, Jordi Planes
    Adversarial Robustness of Deep Learning-based Malware Detectors via (De) Randomized Smoothing
    IEEE Access 2024 - Random Chunks-based (De)Randomized Smoothing
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
            min_len=min_len,
            sort_by_size=sort_by_size,
            file_percentage=file_percentage,
            num_chunks=num_chunks,
            is_training=is_training,
            min_chunk_size=min_chunk_size
        )

    def generate_testing_examples(self, batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        During testing, given an input example we sequentially extract N chunks of bytes. This chunks may overlap with
        one another depending on the amount of chunks you want to extract and the size of each chunk
        """
        if len(batch) == 1:  # Only implemented for batch sizes equals to 1
            # Code here
            x = batch[0][0]
            max_size = x.size()[0]
            chunk_size = int(-(-(max_size * self.file_percentage) // 1))  # Used to round up the float value
            chunk_size = max(self.min_chunk_size, chunk_size) # The chunk size has to be at least equal to the kernel size of the first convolutional layer

            vecs = []
            for _ in range(self.num_chunks):
                curr_idx = randint(0, len(x) - chunk_size)
                vecs.append(x[curr_idx:curr_idx + chunk_size])

            x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True, padding_value=self.padding_idx)
            y = batch[0][1]
            return x, y
        else:
            raise NotImplementedError


