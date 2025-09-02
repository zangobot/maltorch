# Maltorch: Pentesting Suite for AI-based Windows Malware Detectors
![PyPI](https://img.shields.io/pypi/v/maltorch?style=flat-square)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/zangobot/maltorch?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/zangobot/maltorch?style=flat-square)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/maltorch?style=flat-square)
![PyPI - Downloads](https://img.shields.io/pypi/dm/maltorch?style=flat-square)

Testing the security of AI-based Windows malware detectors has never been easier!
You can deploy `maltorch`, boot up models already trained with PyTorch, and deploy an arsenal of testing techniques before placing an AV in production.


## Installation
You can install `maltorch` through pip, but you also need a custom version of [EMBER](https://github.com/elastic/ember) due to numpy portability issues.
```bash
pip install maltorch
pip install git+https://github.com/zangobot/ember.git
```

## Loading AI-based Models
The library already provides *tons* of pre-trained models, you can instantiate one by just:
```python
from maltorch.zoo.malconv import MalConv
model = MalConv.create_model()
```
and it also accepts `device` parameter to load the model in GPU.

## Evasion Attacks
Straight-forward way to compute attacks!
You just neet to load the model, instantiate the attack, and then pass the model to the freshly-created technique:
```python
from torch.utils.data import TensorDataset, DataLoader
from maltorch.adv.evasion.partialdos import PartialDOS
from maltorch.data.loader import load_from_folder, create_labels
from maltorch.zoo.malconv import MalConv

model = MalConv.create_model()

# Load data as a Pytorch DataLoader
folder_with_exe = ...
X = load_from_folder(folder_with_exe, "exe",device=device)
y = create_labels(X, 1, device=device)
dl = DataLoader(TensorDataset(X, y), batch_size=3)

# Store adversarial EXEmples into a new DataLoader
attack = PartialDOS(query_budget=3)
adversarial_loader = attack(model, torch_data_loader)
```
