import torch
import random


class RandomChunkSampler(torch.utils.data.sampler.Sampler):
    """
    Samples random "chunks" of a dataset, so that items within a chunk are always loaded together. Useful to keep chunks in similar size groups to reduce runtime.
    """

    def __init__(self, data_source, batch_size):
        """
        data_source: the souce pytorch dataset object
        batch_size: the size of the chunks to keep together. Should generally be set to the desired batch size during training to minimize runtime.
        """
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        n = len(self.data_source)

        data = [x for x in range(n)]

        # Create blocks
        blocks = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        # shuffle the blocks
        random.shuffle(blocks)
        # concatenate the shuffled blocks
        data[:] = [b for bs in blocks for b in bs]

        return iter(data)

    def __len__(self):
        return len(self.data_source)