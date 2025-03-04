from maltorch.zoo.resnet18 import ResNet18
from maltorch.datasets.grayscale_dataset import GrayscaleDataset
from torch.utils.data import DataLoader
from secmlt.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
import torch
import multiprocessing

resnet = ResNet18()

training_dataset = GrayscaleDataset(
    goodware_directory="path/to/goodware/train/",
    malware_directory="path/to/malware/train/"
)


num_workers = max(multiprocessing.cpu_count() - 4, multiprocessing.cpu_count() // 2 + 1)
dataloader = DataLoader(
    training_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers
)
loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(resnet.parameters())
trainer = BasePyTorchTrainer(
    optimizer=optimizer,
    epochs=1,
    loss=loss,
)
trainer.train(resnet, dataloader)

