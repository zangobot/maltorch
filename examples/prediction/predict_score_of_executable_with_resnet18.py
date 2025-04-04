
import torch
from maltorch.zoo.resnet18 import ResNet18
from maltorch.data.loader import load_single_exe
from maltorch.data_processing.grayscale_preprocessing import GrayscalePreprocessing

exe_filepath = "path/to/exe/file/"
model_path = "/path/to/model/state/dict"

preprocessing = GrayscalePreprocessing(
    width=256,
    height=256,
    convert_to_3d_image=True
)

classifier = ResNet18.create_model(
    model_path=model_path,
    preprocessing=preprocessing,
)
x = load_single_exe(exe_filepath).to(torch.long).unsqueeze(0)
print(classifier.predict(x).item())
