from typing import Union, List, Type, Callable

from secmlt.trackers.trackers import Tracker
from torch.nn import BCEWithLogitsLoss

from maltorch.adv.evasion.base_optim_attack_creator import (
    BaseOptimAttackCreator,
    OptimizerBackends,
)
from maltorch.adv.evasion.gradfree_attack import GradientFreeBackendAttack
from maltorch.adv.evasion.gradient_attack import GradientBackendAttack
from maltorch.initializers.padding_initializer import PaddingInitializer
from maltorch.manipulations.replacement_manipulation import (
    ReplacementManipulation,
)
from maltorch.optim.optimizer_factory import MalwareOptimizerFactory


class PaddingGradFree(GradientFreeBackendAttack):
    def __init__(
            self,
            query_budget: int,
            padding: int = 2048,
            y_target: Union[int, None] = None,
            population_size: int = 10,
            random_init: bool = False,
            trackers: Union[List[Tracker], Tracker] = None,
    ):
        loss_function = BCEWithLogitsLoss(reduction="none")
        initializer = PaddingInitializer(random_init=random_init, padding=padding)
        manipulation_function = ReplacementManipulation(initializer=initializer)
        optimizer_cls = MalwareOptimizerFactory.create_ga(
            population_size=population_size
        )
        super().__init__(
            y_target=y_target,
            query_budget=query_budget,
            loss_function=loss_function,
            optimizer_cls=optimizer_cls,
            manipulation_function=manipulation_function,
            initializer=initializer,
            trackers=trackers,
        )


class PaddingGrad(GradientBackendAttack):
    def __init__(
            self,
            query_budget: int,
            padding: int = 2048,
            y_target: Union[int, None] = None,
            random_init: bool = False,
            step_size: int = 58,
            device: str = "cpu",
            trackers: Union[List[Tracker], Tracker] = None,
    ):
        loss_function = BCEWithLogitsLoss(reduction="none")
        initializer = PaddingInitializer(random_init=random_init, padding=padding)
        manipulation_function = ReplacementManipulation(initializer=initializer)
        optimizer_cls = MalwareOptimizerFactory.create_bgd(lr=step_size, device=device)
        super().__init__(
            y_target=y_target,
            query_budget=query_budget,
            loss_function=loss_function,
            optimizer_cls=optimizer_cls,
            manipulation_function=manipulation_function,
            initializer=initializer,
            trackers=trackers,
        )


class Padding(BaseOptimAttackCreator):
    @staticmethod
    def get_backends() -> set[str]:
        return {OptimizerBackends.GRADIENT, OptimizerBackends.NG}

    @staticmethod
    def _get_nevergrad_implementation() -> Type[PaddingGradFree]:
        return PaddingGradFree

    @staticmethod
    def _get_native_implementation() -> Type[PaddingGrad]:
        return PaddingGrad

    def __new__(
            cls,
            query_budget: int,
            padding: int = 2048,
            y_target: Union[int, None] = None,
            random_init: bool = False,
            step_size: int = 16,
            population_size: int = 10,
            device: str = "cpu",
            trackers: Union[List[Tracker], Tracker] = None,
            backend: str = OptimizerBackends.GRADIENT,
    ) -> Callable:
        implementation: Callable = cls.get_implementation(backend)
        if backend == OptimizerBackends.GRADIENT:
            kwargs = {"step_size": step_size, "device": device}
        else:
            kwargs = {
                "population_size": population_size,
            }
        return implementation(
            padding=padding,
            query_budget=query_budget,
            y_target=y_target,
            trackers=trackers,
            random_init=random_init,
            **kwargs,
        )
