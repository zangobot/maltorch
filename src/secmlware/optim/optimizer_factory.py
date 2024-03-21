from functools import partial
from typing import Union, Type

import nevergrad
from nevergrad.optimization.differentialevolution import DE, TwoPointsDE, DifferentialEvolution

from secmlware.optim.base import BaseByteOptimizer
from secmlware.optim.bgd import BGD
from secmlware.optim.byte_gradient_processing import ByteGradientProcessing

TORCH_OPTIM_TYPE = partial[BaseByteOptimizer]
NEVERGRAD_OPTIM_TYPE = partial[nevergrad.optimization.Optimizer]
OPTIM_TYPE = Union[TORCH_OPTIM_TYPE, NEVERGRAD_OPTIM_TYPE]


class MalwareOptimizerFactory:

    @staticmethod
    def create(optim_cls: Union[str, OPTIM_TYPE], **optimizer_args):
        if type(optim_cls) is not str:
            return partial(optim_cls, **optimizer_args)()
        if optim_cls == "bgd":
            return MalwareOptimizerFactory.create_bgd(**optimizer_args)
        if optim_cls == "ga":
            return MalwareOptimizerFactory.create_ga()
        raise NotImplementedError(f"Optimizer {optim_cls} not included.")

    @staticmethod
    def create_bgd(lr: int, device: str = "cpu"):
        return partial(
            BGD,
            lr=lr,
            gradient_processing=ByteGradientProcessing(),
            device=device,
        )

    @staticmethod
    def create_ga():
        return DifferentialEvolution(popsize=10, crossover="twopoints")
