from pathlib import Path
from typing import Type, Union, List, Callable

import nevergrad
from nevergrad.optimization import Optimizer
from secmlt.models.base_model import BaseModel

from secmlt.trackers import Tracker
from torch.nn import BCEWithLogitsLoss, BCELoss

from maltorch.adv.evasion.base_optim_attack_creator import (
    BaseOptimAttackCreator,
    OptimizerBackends,
)
from maltorch.adv.evasion.gradfree_attack import GradientFreeBackendAttack
from maltorch.initializers.initializers import IdentityInitializer
from maltorch.manipulations.gamma_section_injection_manipulation import GAMMASectionInjectionManipulation
from maltorch.optim.optimizer_factory import MalwareOptimizerFactory


class GAMMASectionInjectionGradFree(GradientFreeBackendAttack):
    def __init__(
            self,
            query_budget: int,
            benignware_folder: Path,
            how_many_sections: int,
            which_sections: list = None,
            y_target: Union[int, None] = None,
            population_size: int = 10,
            random_init: bool = False,
            model_outputs_logits: bool = True,
            device: str = "cpu",
            reg_parameter: float = 1e-5,
            trackers: Union[List[Tracker], Tracker] = None,
    ):
        if which_sections is None:
            which_sections = ['rodata']
        initializer = IdentityInitializer(
            random_init=random_init
        )
        optimizer_cls = MalwareOptimizerFactory.create_ga(
            population_size=population_size
        )
        loss_function = BCEWithLogitsLoss(reduction="none") if model_outputs_logits else BCELoss(reduction="none")
        manipulation_function = GAMMASectionInjectionManipulation(benignware_folder=benignware_folder,
                                                                  which_sections=which_sections,
                                                                  how_many_sections=how_many_sections)
        super().__init__(
            y_target=y_target,
            query_budget=query_budget,
            loss_function=loss_function,
            initializer=initializer,
            manipulation_function=manipulation_function,
            optimizer_cls=optimizer_cls,
            trackers=trackers,
            device=device,
            reg_parameter=reg_parameter
        )

    def _init_optimizer(self, model: BaseModel, delta: nevergrad.p.Array) -> Optimizer:
        self.optimizer = self.optimizer_cls(
            parametrization=nevergrad.p.Array(
                shape=delta.value.shape, lower=0.0, upper=1.0
            ), budget=self.query_budget
        )
        return self.optimizer


class GAMMASectionInjection(BaseOptimAttackCreator):
    """
    GAMMA Section Injection attack

    Demetrio, L., Biggio, B., Lagorio, G., Roli, F., & Armando, A. (2021).
    Functionality-preserving black-box optimization of adversarial windows malware.
    IEEE Transactions on Information Forensics and Security, 16, 3469-3478.
    """

    @staticmethod
    def get_backends() -> set[str]:
        return {OptimizerBackends.NG}

    @staticmethod
    def _get_nevergrad_implementation() -> Type[GAMMASectionInjectionGradFree]:
        return GAMMASectionInjectionGradFree

    def __new__(
            cls,
            query_budget: int,
            benignware_folder: Path,
            which_sections: list = None,
            y_target: Union[int, None] = None,
            how_many_sections: int = 75,
            random_init: bool = False,
            population_size: int = 10,
            reg_parameter: float = 1e-5,
            device: str = "cpu",
            model_outputs_logits: bool = True,
            trackers: Union[List[Tracker], Tracker] = None,
            backend: str = OptimizerBackends.NG,
    ) -> Callable:
        if which_sections is None:
            which_sections = ['rodata']
        return GAMMASectionInjectionGradFree(
            query_budget=query_budget,
            benignware_folder=benignware_folder,
            how_many_sections=how_many_sections,
            which_sections=which_sections,
            population_size=population_size,
            y_target=y_target,
            trackers=trackers,
            random_init=random_init,
            model_outputs_logits=model_outputs_logits,
            device=device,
            reg_parameter=reg_parameter
        )
