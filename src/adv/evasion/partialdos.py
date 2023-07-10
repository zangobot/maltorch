from typing import List

import torch
from secml2.models.base_model import BaseModel
from secml2.optimization.constraints import Constraint
from torch.nn import CrossEntropyLoss

from src.adv.evasion.composite import MalwareCompositeEvasionAttack
from src.manipulations.replacement import ReplacementManipulation
from src.optim.bgd import BGD
from src.optim.byte_gradient_processing import ByteGradientProcessing
from src.optim.initializers import ByteBasedInitializer


class PartialDOS(MalwareCompositeEvasionAttack):
    def __init__(
        self,
        num_steps: int,
        step_size: int,
    ):
        loss_function = CrossEntropyLoss()
        optimizer_cls = BGD
        self.manipulation = torch.LongTensor(list(range(2, 58)))
        manipulation_function = ReplacementManipulation(self.manipulation)
        domain_constraints = []
        perturbation_constraints = []
        initializer = ByteBasedInitializer(56)
        super().__init__(
            None,
            num_steps,
            step_size,
            loss_function,
            optimizer_cls,
            manipulation_function,
            domain_constraints,
            perturbation_constraints,
            initializer,
            gradient_processing=ByteGradientProcessing(),
        )

    def init_perturbation_constraints(self) -> List[Constraint]:
        return []

    def create_optimizer(self, delta: torch.Tensor, model: BaseModel):
        optimizer = self.optimizer_cls(
            params=[delta],
            model=model,
            indexes_to_perturb=self.manipulation.repeat((delta.shape[0], 1)),
            gradient_processing=self.gradient_processing,
            lr=self.step_size,
            device="cpu",
        )
        return optimizer
