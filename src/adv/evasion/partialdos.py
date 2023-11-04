import torch
from secml2.models.base_model import BaseModel
from torch.nn import CrossEntropyLoss

from src.adv.evasion.composite import MalwareCompositeEvasionAttack
from src.manipulations.replacement import ReplacementManipulation
from src.optim.byte_gradient_processing import ByteGradientProcessing
from src.optim.initializers import ByteBasedInitializer, PartialDOSInitializer
from src.optim.optimizer_factory import MalwareOptimizerFactory


class PartialDOS(MalwareCompositeEvasionAttack):
    """
    Partial DOS attack.

    Demetrio, L., Biggio, B., Giovanni, L., Roli, F., & Alessandro, A. (2019).
    Explaining vulnerabilities of deep learning to adversarial malware binaries.
    In CEUR WORKSHOP PROCEEDINGS (Vol. 2315).
    """

    def __init__(
        self,
        num_steps: int,
        step_size: int,
        random_init: bool = False,
        device: str = "cpu",
    ):
        """
        Create the Partial DOS attack.
        :param num_steps: Number of optimization steps.
        :param step_size: Number of byte modified at each iteration.
        :param random_init: Initialize the manipulation with 0 or with random bytes.
        :param device: Device to use for computation.
        """
        loss_function = CrossEntropyLoss()
        self.manipulation = torch.LongTensor(list(range(2, 58)))
        optimizer_cls = MalwareOptimizerFactory.create_bgd(
            lr=step_size, device=device, indexes_to_perturb=self.manipulation
        )
        initializer = PartialDOSInitializer(random_init=random_init)
        manipulation_function = ReplacementManipulation(initializer=initializer)
        domain_constraints = []
        perturbation_constraints = []
        super().__init__(
            y_target=None,
            num_steps=num_steps,
            step_size=step_size,
            loss_function=loss_function,
            optimizer_cls=optimizer_cls,
            manipulation_function=manipulation_function,
            domain_constraints=domain_constraints,
            perturbation_constraints=perturbation_constraints,
            initializer=initializer,
            gradient_processing=ByteGradientProcessing(),
        )
