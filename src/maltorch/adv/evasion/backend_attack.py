from typing import Union, List, Callable

import torch
from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack
from secmlt.models.base_model import BaseModel
from secmlt.optimization.initializer import Initializer
from secmlt.trackers import Tracker
from torch.utils.data import TensorDataset, DataLoader

from maltorch.manipulations.byte_manipulation import ByteManipulation


class BackendAttack(BaseEvasionAttack):
    @classmethod
    def _trackers_allowed(cls):
        return True

    @staticmethod
    def get_perturbation_models():
        pass

    def __init__(
        self,
        y_target: Union[int, None],
        query_budget: int,
        loss_function: Union[str, torch.nn.Module],
        optimizer_cls: Callable,
        manipulation_function: ByteManipulation,
        initializer: Initializer,
        trackers: Union[List[Tracker], Tracker] = None,
        **kwargs
    ):
        self.y_target = y_target
        self.query_budget = query_budget
        self.loss_function = loss_function
        self.manipulation_function = manipulation_function
        self.initializer = initializer
        self.trackers = trackers
        self.optimizer = None
        self.optimizer_cls = optimizer_cls
        self._best_loss = None
        self._best_delta = None

    def _init_attack_manipulation(
        self, samples: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        return  self.manipulation_function.initialize(samples.data)


    def _apply_manipulation(
        self, x: torch.Tensor, delta: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError()

    def _optimizer_step(self, delta: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _init_optimizer(self, model: BaseModel, delta: torch.Tensor) -> Callable:
        raise NotImplementedError()

    def _consumed_budget(self):
        raise NotImplementedError()

    def _init_best_tracking(self, delta: torch.Tensor):
        self._best_delta = torch.zeros_like(delta.detach().cpu())
        self._best_loss = torch.zeros((delta.shape[0], 1)).fill_(torch.inf)

    def _track_best(self, loss: torch.Tensor, delta: torch.Tensor):
        delta_to_track = delta.cpu()
        where_best = (loss.cpu() < self._best_loss.cpu()).squeeze(1)
        self._best_delta[where_best] = delta_to_track[where_best]
        self._best_loss[where_best] = loss.cpu()[where_best]

    def _get_best_delta(self):
        return self._best_delta

    def _track(
        self,
        iteration: int,
        loss: torch.Tensor,
        scores: torch.Tensor,
        x_adv: torch.Tensor,
        delta: torch.Tensor,
    ):
        if self._trackers_allowed():
            if self.trackers:
                for tracker in self.trackers:
                    tracker.track(
                        iteration=iteration,
                        loss=loss.data,
                        scores=scores.data,
                        x_adv=x_adv.data,
                        delta=delta,
                        grad=None,
                    )

    def __call__(self, model: BaseModel, data_loader: DataLoader) -> DataLoader:
        """
        Compute the attack against the model, using the input data.

        Parameters
        ----------
        model : BaseModel
            Model to test.
        data_loader : DataLoader
            Test dataloader.

        Returns
        -------
        DataLoader
            Dataloader with adversarial examples and original labels.
        """
        adversarials = []
        original_labels = []
        for samples, labels in data_loader:
            x_adv, _ = self._run(model, samples, labels)
            adversarials.append(x_adv)
            original_labels.append(labels)
        adversarials = torch.vstack(adversarials)
        original_labels = torch.vstack(original_labels)
        adversarial_dataset = TensorDataset(adversarials, original_labels)
        return DataLoader(
            adversarial_dataset,
            batch_size=data_loader.batch_size,
        )

    def _run(
        self,
        model: BaseModel,
        samples: torch.Tensor,
        labels: torch.Tensor,
        **optim_kwargs,
    ) -> (torch.Tensor, torch.Tensor):
        multiplier = 1 if self.y_target is not None else -1
        target = (
            torch.zeros_like(labels) + self.y_target
            if self.y_target is not None
            else labels
        ).type(labels.dtype)
        target = target.to(labels.device)
        samples, delta = self._init_attack_manipulation(samples)
        self.optimizer = self._init_optimizer(model, delta)
        budget = 0
        self._init_best_tracking(delta)
        while budget < self.query_budget:
            x_adv, delta = self._apply_manipulation(samples, delta)
            scores = model.decision_function(x_adv)
            loss = self.loss_function(scores, target) * multiplier
            delta = self._optimizer_step(delta, loss)
            budget += self._consumed_budget()
            self._track(budget, loss, scores, x_adv, delta)
            self._track_best(loss, delta)
        best_delta = self._get_best_delta()
        best_x, _ = self._apply_manipulation(samples, best_delta)
        return best_x, self._best_delta
