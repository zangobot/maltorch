from maltorch.zoo.ngramconv import NGramConv
from maltorch.datasets.binary_dataset import BinaryDataset
from torch.utils.data import DataLoader
from maltorch.trainers.binary_trainer import BinaryTrainer
import multiprocessing

ngramconv = NGramConv(out_channels=50)
training_dataset = BinaryDataset(
    goodware_directory="path/to/goodware/train/",
    malware_directory="path/to/malware/train/",
    padding_idx=256,
    max_len=1000000
)
validation_dataset = BinaryDataset(
    goodware_directory="path/to/goodware/validation/",
    malware_directory="path/to/malware/validation/",
    padding_idx=256,
    max_len=1000000
)

num_workers = max(multiprocessing.cpu_count() - 4, multiprocessing.cpu_count() // 2 + 1)
train_dataloader = DataLoader(
    training_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=training_dataset.pad_collate_func)
validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=training_dataset.pad_collate_func
)

trainer = BinaryTrainer(num_epochs=5, patience=2, output_directory_path="models/ngramconv/")
trainer.train(ngramconv, train_dataloader, validation_dataloader)

