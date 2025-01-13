from typing import Type, Union, List, Callable

from secmlt.trackers import Tracker
from torch.nn import BCEWithLogitsLoss

from maltorch.adv.evasion.base_optim_attack_creator import (
    BaseOptimAttackCreator,
    OptimizerBackends,
)
from maltorch.adv.evasion.gradfree_attack import GradientFreeBackendAttack
from maltorch.adv.evasion.gradient_attack import GradientBackendAttack
from maltorch.initializers.section_injection_initializer import SectionInjectionInitializer
from maltorch.manipulations.replacement_manipulation import ReplacementManipulation
from maltorch.optim.optimizer_factory import MalwareOptimizerFactory


class SectionInjectionGradFree(GradientFreeBackendAttack):
    def __init__(
            self,
            query_budget: int,
            how_many_sections: int,
            size_per_section: int,
            y_target: Union[int, None] = None,
            population_size: int = 10,
            random_init: bool = False,
            trackers: Union[List[Tracker], Tracker] = None,
    ):
        initializer = SectionInjectionInitializer(
            random_init=random_init, how_many_sections=how_many_sections, size_per_section=size_per_section
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


class SectionInjectionGrad(GradientBackendAttack):
    def __init__(
            self,
            query_budget: int,
            how_many_sections: int,
            size_per_section: int,
            y_target: Union[int, None] = None,
            random_init: bool = False,
            step_size: int = 58,
            device: str = "cpu",
            trackers: Union[List[Tracker], Tracker] = None,
    ):
        initializer = SectionInjectionInitializer(
            random_init=random_init, how_many_sections=how_many_sections, size_per_section=size_per_section
        )
        optimizer_cls = MalwareOptimizerFactory.create_bgd(
            lr=step_size,
            device=device
        )
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


class SectionInjection(BaseOptimAttackCreator):
    """
    Section Injection attack

    Demetrio, L., Biggio, B., Lagorio, G., Roli, F., & Armando, A. (2021).
    Functionality-preserving black-box optimization of adversarial windows malware.
    IEEE Transactions on Information Forensics and Security, 16, 3469-3478.
    """

    @staticmethod
    def get_backends() -> set[str]:
        return {OptimizerBackends.GRADIENT, OptimizerBackends.NG}

    @staticmethod
    def _get_nevergrad_implementation() -> Type[SectionInjectionGradFree]:
        return SectionInjectionGradFree

    @staticmethod
    def _get_native_implementation() -> Type[SectionInjectionGrad]:
        return SectionInjectionGrad

    def __new__(
            cls,
            query_budget: int,
            y_target: Union[int, None] = None,
            how_many_sections: int = 75,
            size_per_section: int = 512,
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
            how_many_sections=how_many_sections,
            size_per_section=size_per_section,
            y_target=y_target,
            trackers=trackers,
            random_init=random_init,
            **kwargs,
        )
