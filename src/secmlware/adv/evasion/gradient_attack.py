import torch
from secmlt.models.base_model import BaseModel

from secmlware.adv.evasion.malware_attack import MalwareAttack, DELTA_TYPE
from secmlware.optim.base import BaseByteOptimizer


class GradientMalwareAttack(MalwareAttack):
    def _init_attack_manipulation(
        self, samples: torch.Tensor
    ) -> (torch.Tensor, DELTA_TYPE):
        x_adv, delta = self.manipulation_function.initialize(samples.data)
        delta = delta.to(samples.device)
        delta.requires_grad = True
        return x_adv, delta

    def _apply_manipulation(
        self, x: torch.Tensor, delta: DELTA_TYPE
    ) -> (torch.Tensor, torch.Tensor):
        x.data, delta.data = self.manipulation_function(x.data, delta.data)
        return x, delta

    def _init_optimizer(self, model: BaseModel, delta: DELTA_TYPE) -> BaseByteOptimizer:
        self.optimizer = self.optimizer_cls(
            [delta],
            indexes_to_perturb=self.manipulation_function.indexes_to_perturb,
            model=model,
        )
        return self.optimizer

    def _optimizer_step(self, delta: DELTA_TYPE, loss: torch.Tensor) -> DELTA_TYPE:
        loss.sum().backward()
        self.optimizer.step()
        return delta

    def _get_best_delta(self):
        return

    def _consumed_budget(self):
        return 2
