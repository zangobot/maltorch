from maltorch.zoo.malconv import MalConv
from maltorch.datasets.rsdel_dataset import RandomizedDeletionDataset
from torch.utils.data import DataLoader
from maltorch.trainers.early_stopping_pytorch_trainer import EarlyStoppingPyTorchTrainer
import multiprocessing
import torch


malconv = MalConv()
training_dataset = RandomizedDeletionDataset(
    goodware_directory="path/to/goodware/train/",
    malware_directory="path/to/malware/train/",
    is_training=True,
    pdel=0.97,
    padding_idx=256,
    max_len=1000000
)
validation_dataset = RandomizedDeletionDataset(
    goodware_directory="path/to/goodware/train/",
    malware_directory="path/to/malware/train/",
    is_training=False,
    pdel=0.97,
    padding_idx=256,
    max_len=1000000
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
    batch_size=1,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=validation_dataset.pad_collate_func
)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(malconv.parameters())

trainer = EarlyStoppingPyTorchTrainer(
    optimizer=optimizer,
    epochs=5,
    loss=criterion
)
model = trainer.train(
    malconv,
    train_dataloader,
    validation_dataloader,
    patience=2
)
torch.save(model.state_dict(), "models/rsdel_malconv/model.pth")