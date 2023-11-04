import abc
from functools import partial
from typing import Union, Type, List

import torch
from secml2.adv.evasion.composite_attack import CompositeEvasionAttack
from secml2.models.base_model import BaseModel
from secml2.optimization.constraints import Constraint
from secml2.optimization.gradient_processing import GradientProcessing
from secml2.optimization.initializer import Initializer
from torch.optim import Optimizer

from src.manipulations.replacement import ByteManipulation


class MalwareCompositeEvasionAttack(CompositeEvasionAttack, abc.ABC):
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
    ):
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
        )

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
        **optim_kwargs
    ) -> torch.Tensor:
        multiplier = 1 if self.y_target is not None else -1
        target = (
            torch.zeros_like(labels) + self.y_target
            if self.y_target is not None
            else labels
        ).type(labels.dtype)
        samples.data, delta = self.manipulation_function.initialize(samples.data)
        delta.requires_grad = True
        optim_kwargs = {"model": model}
        optimizer = self.create_optimizer(delta, **optim_kwargs)
        x_adv, delta = self.manipulation_function(samples, delta)

        for i in range(self.num_steps):
            scores = model.decision_function(x_adv)
            target = target.to(scores.device)
            loss = self.loss_function(scores, target)
            loss = loss * multiplier
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            x_adv.data, delta.data = self.manipulation_function(
                samples.data, delta.data
            )
        return x_adv

    def create_optimizer(self, delta: torch.Tensor, **kwargs):
        if "model" not in kwargs:
            raise ValueError("Model needed to instantiate BGD")
        optimizer = self.optimizer_cls(
            params=[delta],
            model=kwargs["model"],
            gradient_processing=self.gradient_processing,
        )
        return optimizer
