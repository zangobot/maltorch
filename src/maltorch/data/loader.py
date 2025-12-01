from pathlib import Path
from typing import Optional, Union

import torch


def load_from_folder(
        path: Path, extension: Optional[str] = None, padding: int = 256, limit=None, device="cpu"
) -> torch.Tensor:
    """Create a torch.Tensor whose rows are all the file with extension specified in input.
    Tensor are padded to match the same size.
    :param path: Folder path
    :param extension: default "exe", filters all the file based on this extension
    :param padding: default 256, pad every tensor with this value to uniform the size
    :param limit: default None, limit the number of loaded file, None for load all folder
    :return: a torch.Tensor containing all the file converted into tensors
    """
    X = []
    if extension is None:
        pattern = "*"
    else:
        pattern = f"*.{extension}"
    for filepath in sorted(path.glob(pattern)):
        x = load_single_exe(filepath)
        if x is not None:
            X.append(x)
            if limit is not None and len(X) >= limit:
                break
    X = torch.nn.utils.rnn.pad_sequence(X, padding_value=padding).transpose(0, 1).long()
    X = X.to(device)
    return X


def create_labels(x: torch.Tensor, label: int, device="cpu"):
    """
    Create the labels for the specified data.
    """
    y = torch.zeros((x.shape[0], 1)) + label
    y = y.to(device)
    return y


def load_single_exe(path: Path) -> Union[torch.Tensor, None]:
    """
    Create a torch.Tensor from the file pointed in the path
    :param path: a pathlib Path
    :return: torch.Tensor containing the bytes of the file as a tensor, None if the file is not an exe
    """
    with open(path, "rb") as h:
        code = h.read()
    if len(code) != 0 and code[:2] == b'MZ':
        return torch.frombuffer(bytearray(code), dtype=torch.uint8).to(torch.float)
    return None
