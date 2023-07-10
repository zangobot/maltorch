import secml2.models.base_model
import torch
from secml2.adv.evasion.composite_attack import CompositeEvasionAttack
from secml2.models.base_model import BaseModel
from torch.utils.data import DataLoader, TensorDataset


class MalwareCompositeEvasionAttack(CompositeEvasionAttack):
    def create_optimizer(self, delta: torch.Tensor, model: BaseModel):
        optimizer = self.optimizer_cls([delta], lr=self.step_size)
        return optimizer

    def __call__(self, model: BaseModel, data_loader: DataLoader) -> DataLoader:
        adversarials = []
        original_labels = []
        multiplier = 1 if self.y_target is not None else -1
        perturbation_constraints = self.init_perturbation_constraints()
        for samples, labels in data_loader:
            target = (
                torch.zeros_like(labels) + self.y_target
                if self.y_target is not None
                else labels
            ).type(labels.dtype)
            delta = self.initializer(samples.data)
            delta.requires_grad = True
            optimizer = self.create_optimizer(delta, model)
            x_adv = self.manipulation_function(samples, delta)
            for i in range(self.num_steps):
                scores = model.decision_function(x_adv)
                target = target.to(scores.device)
                loss = self.loss_function(scores, target)
                loss = loss * multiplier
                optimizer.zero_grad()
                loss.backward()
                # delta.grad.data = self.gradient_processing(delta.grad.data)
                optimizer.step()
                for constraint in perturbation_constraints:
                    delta.data = constraint(delta.data)
                x_adv.data = self.manipulation_function(samples.data, delta.data)
                for constraint in self.domain_constraints:
                    x_adv.data = constraint(x_adv.data)
                delta.data = self.manipulation_function.invert(samples.data, x_adv.data)

            adversarials.append(x_adv)
            original_labels.append(labels)

        adversarials = torch.vstack(adversarials)
        original_labels = torch.hstack(original_labels)
        adversarial_dataset = TensorDataset(adversarials, original_labels)
        adversarial_loader = DataLoader(
            adversarial_dataset, batch_size=data_loader.batch_size
        )
        return adversarial_loader
