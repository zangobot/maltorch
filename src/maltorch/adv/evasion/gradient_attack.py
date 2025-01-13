import torch
from secmlt.models.base_model import BaseModel

from maltorch.adv.evasion.backend_attack import BackendAttack
from maltorch.optim.base import BaseByteOptimizer


class GradientBackendAttack(BackendAttack):
    def _init_attack_manipulation(
        self, samples: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        x_adv, delta = self.manipulation_function.initialize(samples.data)
        delta = delta.to(samples.device)
        delta.requires_grad = True
        return x_adv, delta

    def _apply_manipulation(
        self, x: torch.Tensor, delta: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        x.data, delta.data = self.manipulation_function(x.data, delta.data)
        return x, delta

    def _init_optimizer(
        self, model: BaseModel, delta: torch.Tensor
    ) -> BaseByteOptimizer:
        self.optimizer = self.optimizer_cls(
            [delta],
            indexes_to_perturb=self.manipulation_function.indexes_to_perturb,
            model=model,
        )
        return self.optimizer

    def _optimizer_step(self, delta: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        loss.sum().backward()
        self.optimizer.step()
        return delta

    def _consumed_budget(self):
        return 2
