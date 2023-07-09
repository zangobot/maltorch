import torch
from secml2.adv.evasion.composite_attack import CompositeEvasionAttack
from secml2.optimization.initializer import Initializer
from torch.nn import CrossEntropyLoss

from src.manipulations.replacement import ReplacementManipulation
from src.optim.bgd import BGD
from src.optim.byte_gradient_processing import ByteGradientProcessing


class PartialDOS(CompositeEvasionAttack):
    def __init__(
        self,
        num_steps: int,
        step_size: int,
    ):
        loss_function = CrossEntropyLoss()
        optimizer_cls = BGD
        manipulation_function = ReplacementManipulation(
            torch.Tensor(list(range(2, 58)))
        )
        domain_constraints = []
        perturbation_constraints = []
        initializer = Initializer()
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
