from maltorch.zoo.resnet18 import ResNet18
from maltorch.datasets.grayscale_dataset import GrayscaleDataset
from torch.utils.data import DataLoader
from maltorch.trainers.early_stopping_pytorch_trainer import EarlyStoppingPyTorchTrainer
import torch
import multiprocessing


resnet = ResNet18()
training_dataset = GrayscaleDataset(
    goodware_directory="path/to/goodware/train/",
    malware_directory="path/to/malware/train/"
)

validation_dataset = GrayscaleDataset(
    goodware_directory="path/to/goodware/train/",
    malware_directory="path/to/malware/train/"
)

num_workers = max(multiprocessing.cpu_count() - 4, multiprocessing.cpu_count() // 2 + 1)
training_dataloader = DataLoader(
    training_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers
)
validation_dataloader = DataLoader(
    training_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers
)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(resnet.parameters())

trainer = EarlyStoppingPyTorchTrainer(
    optimizer=optimizer,
    epochs=5,
    loss=criterion
)
model = trainer.train(
    resnet,
    training_dataloader,
    validation_dataloader,
    patience=2
)
torch.save(model.state_dict(), "models/resnet18/model.pth")


