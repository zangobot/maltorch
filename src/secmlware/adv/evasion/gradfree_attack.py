import nevergrad
import numpy as np
import torch
from nevergrad.optimization import Optimizer
from secmlt.models.base_model import BaseModel

from secmlware.adv.evasion.malware_attack import MalwareAttack, DELTA_TYPE


class GradientFreeMalwareAttack(MalwareAttack):

    def _optimizer_step(self, delta: DELTA_TYPE, loss: torch.Tensor) -> DELTA_TYPE:
        self.optimizer.tell(delta, loss.item())
        delta = self.optimizer.ask()
        return delta

    def _apply_manipulation(
            self, x: torch.Tensor, delta: DELTA_TYPE
    ) -> (torch.Tensor, torch.Tensor):
        p_delta = torch.from_numpy(delta.value)
        return self.manipulation_function(x.data, p_delta)

    def _apply_constraints(self, delta: DELTA_TYPE) -> DELTA_TYPE:
        for constraint in self.perturbation_constraints:
            delta.value = constraint(delta.value)
        return delta

    def _consumed_budget(self):
        return 1

    def _get_best_delta(self):
        return self.optimizer.provide_recommendation()

    def _init_attack_manipulation(
            self, samples: torch.Tensor
    ) -> (torch.Tensor, DELTA_TYPE):
        x_adv, delta = super()._init_attack_manipulation(samples)
        optim_delta = nevergrad.p.Array(shape=delta.shape, lower=0.0, upper=255.0)
        optim_delta.value = delta.numpy()
        return x_adv, optim_delta

    def _init_optimizer(self, model: BaseModel, delta: DELTA_TYPE) -> Optimizer:
        self.optimizer = self.optimizer_cls(
            parametrization=nevergrad.p.Array(shape=delta.value.shape, lower=0.0, upper=255.0)
        )
        return self.optimizer

    def _run(
            self,
            model: BaseModel,
            samples: torch.Tensor,
            labels: torch.Tensor,
            **optim_kwargs,
    ) -> (torch.Tensor, DELTA_TYPE):
        all_adv = torch.zeros_like(samples)
        all_deltas = []
        for i, (sample, label) in enumerate(zip(samples, labels)):
            x_adv, optim_delta = super()._run(model, sample.unsqueeze(0), label.unsqueeze(0), **optim_kwargs)
            all_adv[i] = x_adv
            all_deltas.append(optim_delta)
        return all_adv, torch.Tensor(np.array([d.value for d in all_deltas]))
