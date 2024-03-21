from typing import Union, List

import torch
from secmlt.trackers.trackers import Tracker
from torch.nn import BCEWithLogitsLoss

from secmlware.adv.evasion.gradfree_attack import GradientFreeMalwareAttack
from secmlware.adv.evasion.gradient_attack import GradientMalwareAttack
from secmlware.manipulations.replacement import (
    ReplacementManipulation,
)
from secmlware.optim.initializers import (
    PartialDOSInitializer,
)
from secmlware.optim.optimizer_factory import MalwareOptimizerFactory


class PartialDOS:
    def __new__(
            cls,
            query_budget: int,
            random_init: bool = False,
            step_size: int = 16,
            y_target: Union[int, None] = None,
            device="cpu",
            loss_function: Union[torch.nn.Module] = BCEWithLogitsLoss(reduction="none"),
            trackers: Union[List[Tracker], Tracker] = None,
    ):
        initializer = PartialDOSInitializer(random_init=random_init)
        manipulation_function = ReplacementManipulation(initializer=initializer)
        perturbation_constraints = []
        optimizer_cls = MalwareOptimizerFactory.create_bgd(lr=step_size)
        return GradientMalwareAttack(
            y_target=y_target,
            query_budget=query_budget,
            loss_function=loss_function,
            optimizer_cls=optimizer_cls,
            manipulation_function=manipulation_function,
            perturbation_constraints=perturbation_constraints,
            initializer=initializer,
            trackers=trackers
        )


class PartialDOSGradFree:
    def __new__(
            cls,
            query_budget: int,
            random_init: bool = False,
            y_target: Union[int, None] = None,
            loss_function: Union[torch.nn.Module] = BCEWithLogitsLoss(reduction="none"),
            trackers: Union[List[Tracker], Tracker] = None,
    ):
        initializer = PartialDOSInitializer(random_init=random_init)
        manipulation_function = ReplacementManipulation(initializer=initializer)
        perturbation_constraints = []
        optimizer_cls = MalwareOptimizerFactory.create_ga()
        return GradientFreeMalwareAttack(
            y_target=y_target,
            query_budget=query_budget,
            loss_function=loss_function,
            optimizer_cls=optimizer_cls,
            manipulation_function=manipulation_function,
            perturbation_constraints=perturbation_constraints,
            initializer=initializer,
            trackers=trackers
        )
