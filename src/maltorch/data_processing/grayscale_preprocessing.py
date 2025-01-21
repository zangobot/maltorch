import torch
from secmlt.models.data_processing.data_processing import DataProcessing
from torchvision.transforms import Resize, Compose, Normalize, PILToTensor, ConvertImageDtype, Grayscale
from maltorch.datasets.grayscale_dataset import get_size
from PIL import Image


class GrayscalePreprocessing(DataProcessing):
    """
    Convert image to grayscale.
    """

    def __init__(self, width=256, height=256, convert_to_3d_image=True):
        super().__init__()
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

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(-1).tolist()
        width, height = get_size(len(x_flat))

        image = Image.new("L", (width, height))
        image.putdata(x_flat)
        image = self.preprocess(image)
        image = torch.cat([image, image, image])
        return torch.unsqueeze(image, 0)
        #return image

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverted transform (if defined).

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            The samples in the input space before the transformation.
        """
        raise NotImplementedError