import torch
from torch.nn import CrossEntropyLoss

from src.adv.evasion.composite import MalwareCompositeEvasionAttack
from src.manipulations.replacement import ReplacementManipulation
from src.optim.initializers import ContentShiftInitializer
from src.optim.optimizer_factory import MalwareOptimizerFactory


class ContentShift(MalwareCompositeEvasionAttack):
    """
    Content Shift attack

    Demetrio, L., Coull, S. E., Biggio, B., Lagorio, G., Armando, A., & Roli, F. (2021).
    Adversarial EXEmples: A survey and experimental evaluation of practical attacks on machine learning for windows malware detection.
    ACM Transactions on Privacy and Security (TOPS), 24(4), 1-31.
    """

    def __init__(
        self,
        preferred_manipulation_size: int,
        num_steps: int,
        step_size: int,
        random_init: bool = False,
        device: str = "cpu",
    ):
        """
        Create the Content Shift attack.
        :param preferred_manipulation_size: Size to be injected, it will be rounded to the nearest file_alignment specified by samples
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
        initializer = ContentShiftInitializer(
            preferred_manipulation_size=preferred_manipulation_size,
            random_init=random_init,
        )
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
