from functools import partial
from typing import Union, Callable

from nevergrad.optimization.differentialevolution import (
    DifferentialEvolution,
)

from maltorch.optim.bgd import BGD
from maltorch.optim.byte_gradient_processing import ByteGradientProcessing


class MalwareOptimizerFactory:
    @staticmethod
    def create(optim_cls: Union[str, Callable], **optimizer_args):
        if not isinstance(optim_cls, str):
            return partial(optim_cls, **optimizer_args)()
        if optim_cls == "bgd":
            return MalwareOptimizerFactory.create_bgd(**optimizer_args)
        if optim_cls == "ga":
            return MalwareOptimizerFactory.create_ga()
        raise NotImplementedError(f"Optimizer {optim_cls} not included.")

    @staticmethod
    def create_bgd(lr: int, device: str = "cpu") -> partial[BGD]:
        return partial(
            BGD,
            lr=lr,
            gradient_processing=ByteGradientProcessing(),
            device=device,
        )

    @staticmethod
    def create_ga(population_size: int = 10) -> DifferentialEvolution:
        return DifferentialEvolution(popsize=population_size, crossover="twopoints")
