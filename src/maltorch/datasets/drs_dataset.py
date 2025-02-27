import random
import torch
from maltorch.datasets.binary_dataset import BinaryDataset

class DeRandomizedSmoothingDataset(BinaryDataset):
    """
    Daniel Gibert, Giulio Zizzo, Quan Le
    Certified Robustness of Static Deep Learning-based Malware Detectors against Patch and Append Attacks
    AISec'23

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
                 chunk_size: int = 512,
                 is_training: bool = True):
        super().__init__(
            csv_filepath=csv_filepath,
            goodware_directory=goodware_directory,
            malware_directory=malware_directory,
            max_len=max_len,
            min_len=min_len,
            padding_idx=padding_idx,
        )
        self.chunk_size = chunk_size
        self.is_training = is_training

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
                if x.shape[0] <= self.chunk_size:
                    vecs.append(x)
                else:  # If the executable's size is greater than the specified chunk size
                    start_location = random.randint(0, max(0, x.shape[0] - self.chunk_size))
                    end_location = start_location + self.chunk_size
                    vecs.append(x[start_location: end_location])
                labels.append(y)
            x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True, padding_value=self.padding_idx)
            # stack will give us (B, 1), so index [:,0] to get to just (B)
            y = torch.tensor(labels)
            return x, y
        else:  # Split an executable into chunks
            if len(batch) == 1: # Only implemented for batch sizes equals to 1
                x = batch[0][0]
                x = [x[i:i+self.chunk_size] for i in range(0, len(x), self.chunk_size)]
                x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.padding_idx)
                y = batch[0][1]
                return x, y
            else:
                raise NotImplementedError
