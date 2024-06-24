from secmlware.zoo.malconv import MalConv
from secmlware.data.binary_dataset import BinaryDataset
from torch.utils.data import DataLoader
from secmlware.data.binary_trainer import BinaryTrainer
import multiprocessing

malconv = MalConv()
training_dataset = BinaryDataset(
    goodware_directory="path/to/goodware/train/",
    malware_directory="path/to/malware/train/"
)
validation_dataset = BinaryDataset(
    goodware_directory="path/to/goodware/train/",
    malware_directory="path/to/malware/train/"
)

num_workers = max(multiprocessing.cpu_count() - 4, multiprocessing.cpu_count() // 2 + 1)
train_dataloader = DataLoader(
    training_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=training_dataset.pad_collate_func)
validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=training_dataset.pad_collate_func
)

trainer = BinaryTrainer(num_epochs=5, patience=2, output_directory_path="models/malconv/")
trainer.train(malconv, train_dataloader, validation_dataloader)

