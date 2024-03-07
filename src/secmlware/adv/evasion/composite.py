import abc
import copy
from functools import partial
from typing import Union, Type, List

import torch
from secmlt.adv.evasion.composite_attack import CompositeEvasionAttack
from secmlt.manipulations.manipulation import Manipulation
from secmlt.models.base_model import BaseModel
from secmlt.optimization.constraints import Constraint
from secmlt.optimization.gradient_processing import GradientProcessing
from secmlt.optimization.initializer import Initializer
from secmlt.trackers.trackers import Tracker
from torch.optim import Optimizer

from secmlware.manipulations.replacement import ByteManipulation
from secmlware.optim.optimizer_factory import MalwareOptimizerFactory


class MalwareCompositeEvasionAttack(CompositeEvasionAttack, abc.ABC):
    """
    Abstract PGD-like optimization algorithm.
    """

    def __init__(
        self,
        y_target: Union[int, None],
        num_steps: int,
        step_size: float,
        loss_function: Union[str, torch.nn.Module],
        optimizer_cls: Union[str, Type[partial[Optimizer]]],
        manipulation_function: ByteManipulation,
        domain_constraints: List[Constraint],
        perturbation_constraints: List[Type[Constraint]],
        initializer: Initializer,
        gradient_processing: GradientProcessing,
        trackers: Union[List[Type[Tracker]], Type[Tracker]] = None,
    ) -> None:
        """
        :param y_target: Target class to reach. None for untargeted.
        :param num_steps: Number of optimization steps.
        :param step_size: Number of byte modified at each iteration.
        :param loss_function: Loss function used to evaluate attack.
        :param optimizer_cls: Optimizer class that will be used. Can be a string.
        :param manipulation_function: Manipulation that will be used during the attack.
        :param domain_constraints: Constraints on the input space.
        :param perturbation_constraints: Constraints on the perturbation.
        :param initializer: Initialization function for the manipulation.
        :param gradient_processing: Processing of gradient, used to normalize its norm.
        :param trackers: Objects that analyze the attack at each iteration.
        """
        super().__init__(
            y_target,
            num_steps,
            step_size,
            loss_function,
            optimizer_cls,
            manipulation_function,
            domain_constraints,
            perturbation_constraints,
            initializer,
            gradient_processing,
            trackers,
        )

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
        **optim_kwargs,
    ) -> torch.Tensor:
        multiplier = 1 if self.y_target is not None else -1
        target = (
            torch.zeros_like(labels) + self.y_target
            if self.y_target is not None
            else labels
        ).type(labels.dtype)
        x, delta = self.manipulation_function.initialize(samples.data)
        delta.requires_grad = True

        optimizer = self.create_optimizer(delta, model=model)
        x_adv, delta = self.manipulation_function(samples, delta)

        for i in range(self.num_steps):
            scores = model.decision_function(x_adv)
            target = target.to(scores.device)
            losses = self.loss_function(scores, target)
            loss = losses.sum() * multiplier
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for constraint in self.perturbation_constraints:
                delta.data = constraint(delta.data)
            x_data_prev = copy.deepcopy(x_adv.data)
            x_adv.data, delta.data = self.manipulation_function(
                samples.data, delta.data
            )
            for constraint in self.domain_constraints:
                x_adv.data = constraint(x_adv.data)
            if self.trackers is not None:
                for tracker in self.trackers:
                    tracker.track(
                        iteration=i,
                        loss=losses.data,
                        scores=scores.data,
                        x_adv=x_adv.data,
                        delta=delta.data,
                        grad=x_adv.data - x_data_prev,
                    )
        return x_adv

    # def create_optimizer(self, delta: torch.Tensor, **kwargs) -> Optimizer:
    #     """
    #     Creates the optimizer for the attack.
    #     :param delta: The manipulation to be fine-tuned.
    #     :param kwargs: Optional parameters that must contains 'model' argument.
    #     :return: Created optimizer.
    #     """
    #     if "model" not in kwargs:
    #         raise ValueError("Model needed to instantiate BGD")
    #     optimizer = self.optimizer_cls(
    #         params=[delta],
    #         model=kwargs["model"],
    #         gradient_processing=self.gradient_processing,
    #     )
    #     return optimizer
