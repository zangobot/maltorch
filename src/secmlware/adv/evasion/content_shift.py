from typing import Union, List, Type

import torch
from secmlt.trackers.trackers import Tracker
from torch.nn import CrossEntropyLoss

from secmlware.adv.evasion.gradient_attack import GradientMalwareAttack
from secmlware.manipulations.replacement import ReplacementManipulation
from secmlware.optim.byte_gradient_processing import ByteGradientProcessing
from secmlware.optim.initializers import ContentShiftInitializer
from secmlware.optim.optimizer_factory import MalwareOptimizerFactory


class ContentShift(GradientMalwareAttack):
    """
    Content Shift attack

    Demetrio, L., Coull, S. E., Biggio, B., Lagorio, G., Armando, A., & Roli, F. (2021).
    Adversarial EXEmples: A survey and experimental evaluation of practical attacks on machine learning for windows malware detection.
    ACM Transactions on Privacy and Security (TOPS), 24(4), 1-31.
    """

    def __init__(
        self,
        preferred_manipulation_size: int,
        query_budget: int,
        step_size: int,
        random_init: bool = False,
        device: str = "cpu",
        trackers: Union[List[Type[Tracker]], Type[Tracker]] = None,
    ):
        """
        Create the Content Shift attack.
        :param preferred_manipulation_size: Size to be injected, it will be rounded to the nearest file_alignment specified by samples
        :param num_steps: Number of optimization steps.
        :param step_size: Number of byte modified at each iteration.
        :param random_init: Initialize the manipulation with 0 or with random bytes.
        :param device: Device to use for computation.
        :param trackers: Optional trackers that provide insights on the computations.
        """
        loss_function = CrossEntropyLoss(reduction="none")
        optimizer_cls = MalwareOptimizerFactory.create_bgd(lr=step_size, device=device)
        initializer = ContentShiftInitializer(
            preferred_manipulation_size=preferred_manipulation_size,
            random_init=random_init,
        )
        self.manipulation = torch.LongTensor(list(range(2, 58)))
        manipulation_function = ReplacementManipulation(initializer=initializer)
        super().__init__(
            y_target=None,
            query_budget=query_budget,
            loss_function=loss_function,
            optimizer_cls=optimizer_cls,
            manipulation_function=manipulation_function,
            initializer=initializer,
            trackers=trackers,
        )
