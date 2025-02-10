from typing import Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, Normalize, PILToTensor, ConvertImageDtype, Grayscale
import torch


def get_size(bytez_length: int) -> Tuple[int, int]:
    if (bytez_length < 10240):
        width = 32
    elif (10240 <= bytez_length <= 10240 * 3):
        width = 64
    elif (10240 * 3 <= bytez_length <= 10240 * 6):
        width = 128
    elif (10240 * 6 <= bytez_length <= 10240 * 10):
        width = 256
    elif (10240 * 10 <= bytez_length <= 10240 * 20):
        width = 384
    elif (10240 * 20 <= bytez_length <= 10240 * 50):
        width = 512
    elif (10240 * 50 <= bytez_length <= 10240 * 100):
        width = 768
    else:
        width = 1024

    height = int(bytez_length / width) + 1
    return width, height

class GrayscaleDataset(Dataset):
    def __init__(self, csv_filepath, width=256, height=256, convert_to_3d_image=True):
        self.all_files = []
        self.width = width
        self.height = height
        self.convert_to_3d_image = convert_to_3d_image
        if self.convert_to_3d_image:
            self.preprocess = Compose(
                [
                    PILToTensor(),
                    ConvertImageDtype(torch.float),
                    Resize((self.width, self.height)),
                    Normalize((0.5), (0.5)),
                    Grayscale(num_output_channels=3)
                ]
            )
        else:
            self.preprocess = Compose(
                [
                    PILToTensor(),
                    ConvertImageDtype(torch.float),
                    Resize((self.width, self.height)),
                    Normalize((0.5), (0.5))
                ]
            )

        with open(csv_filepath, "r") as input_file:
            lines = input_file.readlines()
            for line in lines:
                tokens = line.strip().split(",")
                self.all_files.append([tokens[0], int(tokens[1])])

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        to_load, label = self.all_files[index]
        img = self.convert_to_grayscale_img(to_load)
        return img, torch.tensor(label)

    def convert_to_grayscale_img(self, to_load) -> torch.Tensor:
        with open(to_load, "rb") as input_file:
            bytez = input_file.read()
        width, height = get_size(len(bytez))

        image = Image.new("L", (width,height))
        image.putdata(bytez)
        image = self.preprocess(image)
        image = torch.cat([image, image, image])

        return image

