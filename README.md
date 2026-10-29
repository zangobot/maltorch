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

otherwise just clone the repository and install all the dependencies with `uv`:
```commandline
git clone https://github.com/zangobot/maltorch.git
cd maltorch
uv venv --python 3.14
uv pip install -r requirements.txt
```

**Apple‑Silicon note:** On macOS with Apple Silicon the pip wheels for PyTorch and LightGBM bundle incompatible OpenMP runtimes, which can cause crashes.  Install the whole environment via `conda‑forge` instead:
```bash
# Create a fresh conda environment (Python 3.12)
conda create -n maltorch python=3.12
conda activate maltorch

# Install all packages that are available on conda‑forge.  These include the heavy ML and OpenMP libraries.
conda install -c conda-forge \
    pytorch torchvision numpy lightgbm gymnasium stable-baselines3 py-lief
```
Remaining dependencies that are not packaged on conda‑forge must be installed with pip
(torch and torchvision are already satisfied by the conda installation above.)
```
pip install -r requirements.txt
```
## Included manipulations
Maltorch contains most of the practical manipulations of this field:
* DOS header extension and editing (PartialDOS, FullDOS, Extend)
* Content shifting
* Section injection with optimized or goodware content (GAMMA)
* Padding

Most of the manipulations can be used either with gradient-based and gradient-free optimization.

## Loading AI-based Models
The library already provides *tons* of pre-trained models, you can instantiate one by just:
```python
from maltorch.zoo.malconv import MalConv
model = MalConv.create_model()
```
and it also accepts `device` parameter to load the model in GPU.

## Evasion Attacks
Straight-forward way to compute attacks!
You just need to load the model, instantiate the attack, and then pass the model to the freshly-created technique:
```python
from torch.utils.data import TensorDataset, DataLoader
from maltorch.adv.evasion.partialdos import PartialDOS
from maltorch.data.loader import load_from_folder, create_labels
from maltorch.zoo.malconv import MalConv

device = 'cpu'

model = MalConv.create_model()

# Load data as a Pytorch DataLoader
folder_with_exe = ...
X = load_from_folder(folder_with_exe, "exe", device=device)
y = create_labels(X, 1, device=device)
dl = DataLoader(TensorDataset(X, y), batch_size=3)

# Store adversarial EXEmples into a new DataLoader
attack = PartialDOS(query_budget=3)
adversarial_loader = attack(model, dl)
```

## Known Issues
**LightGBM (needed for loading GBDT models) & PyTorch OMP conflict on macOS (Apple Silicon)** – When using both libraries in the same process, they load different OpenMP runtimes (`libomp.dylib` vs `libiomp5.dylib`). This leads to crashes such as:
  ```
  OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
  ```
  **Possible Fixes**:
keep the installation that you prefer, but you need to set:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
```
this should solve the issue at the cost of slow inference of trees in case of parallel processing.