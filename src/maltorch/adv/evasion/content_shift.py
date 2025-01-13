from typing import Type, Union, List, Callable

from secmlt.trackers import Tracker
from torch.nn import BCEWithLogitsLoss

from maltorch.adv.evasion.base_optim_attack_creator import (
    BaseOptimAttackCreator,
    OptimizerBackends,
)
from maltorch.adv.evasion.gradfree_attack import GradientFreeBackendAttack
from maltorch.adv.evasion.gradient_attack import GradientBackendAttack
from maltorch.initializers.content_shift_initializer import ContentShiftInitializer
from maltorch.manipulations.replacement_manipulation import ReplacementManipulation
from maltorch.optim.optimizer_factory import MalwareOptimizerFactory


class ContentShiftGradFree(GradientFreeBackendAttack):
    def __init__(
            self,
            query_budget: int,
            manipulation_size: int,
            y_target: Union[int, None] = None,
            population_size: int = 10,
            random_init: bool = False,
            trackers: Union[List[Tracker], Tracker] = None,
    ):
        initializer = ContentShiftInitializer(
            random_init=random_init, preferred_manipulation_size=manipulation_size
        )
        optimizer_cls = MalwareOptimizerFactory.create_ga(
            population_size=population_size
        )
        loss_function = BCEWithLogitsLoss(reduction="none")
        manipulation_function = ReplacementManipulation(initializer=initializer)
        super().__init__(
            y_target=y_target,
            query_budget=query_budget,
            loss_function=loss_function,
            initializer=initializer,
            manipulation_function=manipulation_function,
            optimizer_cls=optimizer_cls,
            trackers=trackers,
        )


class ContentShiftGrad(GradientBackendAttack):
    def __init__(
            self,
            query_budget: int,
            manipulation_size: int,
            y_target: Union[int, None] = None,
            random_init: bool = False,
            step_size: int = 58,
            device: str = "cpu",
            trackers: Union[List[Tracker], Tracker] = None,
    ):
        initializer = ContentShiftInitializer(
            random_init=random_init, preferred_manipulation_size=manipulation_size
        )
        optimizer_cls = MalwareOptimizerFactory.create_bgd(lr=step_size, device=device)
        loss_function = BCEWithLogitsLoss(reduction="none")
        manipulation_function = ReplacementManipulation(initializer=initializer)
        super().__init__(
            y_target=y_target,
            query_budget=query_budget,
            loss_function=loss_function,
            optimizer_cls=optimizer_cls,
            manipulation_function=manipulation_function,
            initializer=initializer,
            trackers=trackers,
        )


class ContentShift(BaseOptimAttackCreator):
    """
    Content Shift attack

    Demetrio, L., Coull, S. E., Biggio, B., Lagorio, G., Armando, A., & Roli, F. (2021).
    Adversarial EXEmples: A survey and experimental evaluation of practical attacks on machine learning for windows malware detection.
        ACM Transactions on Privacy and Security (TOPS), 24(4), 1-31.
    """

    @staticmethod
    def get_backends() -> set[str]:
        return {OptimizerBackends.GRADIENT, OptimizerBackends.NG}

    @staticmethod
    def _get_nevergrad_implementation() -> Type[ContentShiftGradFree]:
        return ContentShiftGradFree

    @staticmethod
    def _get_native_implementation() -> Type[ContentShiftGrad]:
        return ContentShiftGrad

    def __new__(
            cls,
            query_budget: int,
            y_target: Union[int, None] = None,
            perturbation_size: int = 512,
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
            query_budget=query_budget,
            perturbation_size=perturbation_size,
            y_target=y_target,
            trackers=trackers,
            random_init=random_init,
            **kwargs,
        )
