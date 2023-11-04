from functools import partial

from secml2.optimization.optimizer_factory import OptimizerFactory

from src.optim.bgd import BGD
from src.optim.byte_gradient_processing import ByteGradientProcessing


class MalwareOptimizerFactory(OptimizerFactory):
    @staticmethod
    def create_bgd(lr: int, indexes_to_perturb, device: str = "cpu"):
        return partial(
            BGD,
            indexes_to_perturb=indexes_to_perturb,
            lr=lr,
            gradient_processing=ByteGradientProcessing(),
            device=device,
        )
