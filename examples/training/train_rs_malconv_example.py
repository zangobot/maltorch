from maltorch.zoo.malconv import MalConv
from maltorch.datasets.rs_dataset import RandomizedAblationDataset
from torch.utils.data import DataLoader
from maltorch.trainers.smoothing_classifier_trainer import SmoothingClassifierTrainer
import multiprocessing

malconv = MalConv()
training_dataset = RandomizedAblationDataset(
    goodware_directory="path/to/goodware/train/",
    malware_directory="path/to/malware/train/",
    is_training=True,
    pabl=0.80,
    padding_idx=256,
)
validation_dataset = RandomizedAblationDataset(
    goodware_directory="path/to/goodware/train/",
    malware_directory="path/to/malware/train/",
    is_training=False,
    pabl=0.80,
    padding_idx=256,
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

trainer = SmoothingClassifierTrainer(num_epochs=5, patience=2, output_directory_path="models/rs_malconv/")
trainer.train(malconv, train_dataloader, validation_dataloader)

