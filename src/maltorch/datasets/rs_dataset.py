import torch
from maltorch.datasets.binary_dataset import BinaryDataset


class RandomizedAblationDataset(BinaryDataset):
    """
    Daniel Gibert, Giulio Zizzo, Quan Le
    Towards a Practical Defense Against Adversarial Attacks on Deep Learning-Based Malware Detectors via Randomized
    Smoothing.
    ESORICS Workshops, SECAI 2024
    """
    def __init__(self,
                 csv_filepath: str = None,
                 goodware_directory: str = None,
                 malware_directory: str = None,
                 max_len: int = 2**20,
                 padding_idx: int = 256,
                 min_len: int = None,
                 num_versions: int = 100,
                 pabl: float = 0.20,
                 is_training: bool = True):
        super().__init__(
            csv_filepath=csv_filepath,
            goodware_directory=goodware_directory,
            malware_directory=malware_directory,
            max_len=max_len,
            padding_idx=padding_idx,
            min_len=min_len
        )
        self.num_versions = num_versions
        self.pabl = pabl
        self.is_training = is_training

    def pad_collate_func(self, batch):
        """
        This function randomly ablates the bytes given a probability pabl.
        It works differently at training and at test time.
        During training, given an input example we generate an ablated version of that example.
        During testing, given an input example we generate N ablated versions of that example.
        """
        vecs = []
        labels = []
        if self.is_training is True:
            for x, y in batch:
                # Get mask
                mask = torch.rand(x.shape[0]) <= self.pabl
                # Apply mask - Convert masked elements to self.padding_idx
                masked_x = x.masked_fill(mask, self.padding_idx)
                vecs.append(masked_x)
                labels.append(y)
            x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True, padding_value=self.padding_idx)
            # stack will give us (B, 1), so index [:,0] to get to just (B)
            y = torch.tensor(labels)
            return x, y
        else: # Set self.is_training to False whenever you want to generate N versions for each executable.
            if len(batch) == 1:  # Only implemented for batch sizes equals to 1
                x = batch[0][0]
                for i in range(self.num_versions):
                    # Get mask
                    mask = torch.rand(x.shape[0]) <= self.pabl
                    # Apply mask - Convert masked elements to self.padding_idx
                    masked_x = x.masked_fill(mask, self.padding_idx)
                    vecs.append(masked_x)
                x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True, padding_value=self.padding_idx)
                y = batch[0][1]
                return x, y
            else:
                raise NotImplementedError

