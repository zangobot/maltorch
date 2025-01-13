from pathlib import Path

import torch


def load_from_folder(
    path: Path, extension: str = "exe", padding: int = 256, limit=None
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
    for filepath in path.glob(f"*.{extension}"):
        x = load_single_exe(filepath)
        X.append(x)
        if limit is not None and len(X) >= limit:
            break
    X = torch.nn.utils.rnn.pad_sequence(X, padding_value=padding).transpose(0, 1).long()
    return X


def create_labels(x: torch.Tensor, label: int):
    y = torch.zeros((x.shape[0], 1)) + label
    return y


def load_single_exe(path: Path) -> torch.Tensor:
    """
    Create a torch.Tensor from the file pointed in the path
    :param path: a pathlib Path
    :return: torch.Tensor containing the bytes of the file as a tensor
    """
    with open(path, "rb") as h:
        code = h.read()
    x = torch.frombuffer(bytearray(code), dtype=torch.uint8).to(torch.float)
    return x
