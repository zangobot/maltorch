from typing import Union, Callable

import torch
import numpy as np
from gymnasium import spaces, Env
from stable_baselines3.common.vec_env import DummyVecEnv
from secmlt.models.base_model import BaseModel
from maltorch.adv.evasion.backend_attack import BackendAttack
from maltorch.manipulations.sequential_manipulation import SequentialManipulation
from maltorch.optim.base import BaseByteOptimizer

class RLEvasionEnv(Env):
    def __init__(self, samples: torch.Tensor,
                 manipulation_function: SequentialManipulation,
                 model):
        super().__init__()
        self.samples = samples.cpu()
        self.manipulation_function = manipulation_function
        self.model = model
        self.x = None
        dim = samples.shape[1]
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(dim,), dtype=np.uint8)
        self.num_manipulations = manipulation_function.num_manipulations
        self.action_space = spaces.Discrete(self.num_manipulations)
        self.idx = -1

    def reset(self, **kwargs):
        self.idx = (self.idx + 1) % len(self.samples)
        self.x = self.samples[self.idx].clone()
        return self.x, {}

    def step(self, action):
        action = torch.tensor(action).reshape(-1,1)
        self.x.data, _ = self.manipulation_function(self.x.data, action)
        with torch.no_grad():
            logits = self.model(self.x.unsqueeze(0).long())
        score = logits.mean().item()
        reward = -score
        return self.x, reward, False, False, {"score": score}

class ReinforcementLearningBackendAttack(BackendAttack):
    def __init__(self,
                 query_budget: int,
                 optimizer_cls: Callable,
                 manipulation_function: SequentialManipulation,
                 training_budget: int,
                 training_fraction: float = 0.5,
                 device: str = "cpu",
                 **kwargs):
        super().__init__(
            y_target= None,
            query_budget= query_budget,
            loss_function=None,
            optimizer_cls=optimizer_cls,
            manipulation_function= manipulation_function,
            initializer=None,
            device = device,
            reg_parameter = 0,
            trackers = None,
            early_stopping = None,
            **kwargs
        )
        self._policy = None
        self._training_budget = training_budget
        self._training_fraction = training_fraction

    def __call__(self, model, data_loader):
        """Train the policy on a fraction of the samples and then run the
        base attack on the remaining ones.
        """
        # Gather all data into a single tensor
        samples_list = []
        labels_list = []
        for batch_samples, batch_labels in data_loader:
            samples_list.append(batch_samples)
            labels_list.append(batch_labels)
        all_samples = torch.cat(samples_list, dim=0).to(self.device)
        all_labels = torch.cat(labels_list, dim=0).to(self.device)

        # Split into training and remaining sets
        train_size = int(len(all_samples) * self._training_fraction)
        train_samples = all_samples[:train_size]
        test_samples = all_samples[train_size:]
        test_labels = all_labels[train_size:]

        # Train policy on training subset
        env = DummyVecEnv([lambda: RLEvasionEnv(
            train_samples,
            manipulation_function=self.manipulation_function,
            model=model)])

        self.policy = self.optimizer_cls(env=env)
        # Learn with the environment
        self.policy.learn(total_timesteps=self._training_budget)
        self.query_budget -= self._training_budget
        if self.query_budget < 0:
            self.query_budget = 0

        # Run the base attack on the remaining samples using a new dataloader
        test_dataset = torch.utils.data.TensorDataset(test_samples, test_labels)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False
        )
        return super().__call__(model, test_loader)
    
    def _init_optimizer(self, model: BaseModel,
                        delta: torch.Tensor = None, **optim_kwargs) -> BaseByteOptimizer:
        """Return the policy trained earlier in ``__call__``.
        The training logic is executed in ``__call__`` and the trained
        policy is stored on ``self._policy``.
        """
        return self._policy


    def _optimizer_step(self, x_adv:torch.Tensor,
                        delta: torch.Tensor,
                        loss: torch.Tensor) -> torch.Tensor:
            self._policy.eval()
            obs = x_adv.detach().cpu().numpy()
            actions, _ = self._policy.predict(obs, deterministic=True)
            action_tensor = torch.tensor(actions,
                                         dtype=torch.long,
                                         device=delta.device)
            # Append action to existing delta
            new_delta = torch.cat([delta, action_tensor.unsqueeze(0)], dim=1)
            return new_delta

    def _consumed_budget(self):
        return 1