from functools import partial
from typing import Union, Callable

# ----- external libraries ----------------------------------------------------
from nevergrad.optimization.differentialevolution import DifferentialEvolution
from nevergrad.optimization.optimizerlib import NGOpt
from stable_baselines3 import DQN, SAC, PPO

from maltorch.optim.bgd import BGD
from maltorch.optim.byte_gradient_processing import ByteGradientProcessing


class MalwareOptimizerFactory:
    @staticmethod
    def create(optim_cls: Union[str, Callable], **optimizer_args):
        if not isinstance(optim_cls, str):
            return partial(optim_cls, **optimizer_args)()
        if optim_cls == "bgd":
            return MalwareOptimizerFactory.create_bgd(**optimizer_args)
        if optim_cls == "ga":
            return MalwareOptimizerFactory.create_ga(**optimizer_args)
        raise NotImplementedError(f"Optimizer {optim_cls} not included.")

    @staticmethod
    def create_bgd(lr: int, device: str = "cpu") -> partial[BGD]:
        return partial(
            BGD,
            lr=lr,
            gradient_processing=ByteGradientProcessing(),
            device=device,
        )

    @staticmethod
    def create_ga(population_size: int = 10) -> DifferentialEvolution:
        return DifferentialEvolution(
            popsize=population_size, crossover="onepoint"
        )

    @staticmethod
    def create_ngopt() -> partial[NGOpt]:
        return partial(NGOpt)

    @staticmethod
    def create_policy(policy_cls: Callable,
                      env,
                      **policy_kwargs):
        return policy_cls(env, **policy_kwargs)


    @staticmethod
    def create_sac_policy(
        policy: str = "MlpPolicy",
        verbose: int = 0,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,**kwargs
    ):
        """
        Returns a partial that creates a SAC policy when an env is provided.
        """
        return partial(
            SAC,
            policy=policy,
            verbose=verbose,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            **kwargs
        )

    @staticmethod
    def create_ppo_policy(
        policy: str = "MlpPolicy",
        verbose: int = 0,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,**kwargs
    ):
        """
        Returns a partial that creates a PPO policy when an env is provided.
        """
        return partial(
            PPO,
            policy=policy,
            verbose=verbose,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            **kwargs
        )

    @staticmethod
    def create_dqn_policy(
        policy: str = "MlpPolicy",
        verbose: int = 0,
        learning_rate: float = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 1000,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Returns a factory that creates a DQN policy when an env is provided.
        """
        return partial(
            DQN,
            policy=policy,
            verbose=verbose,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size
        )
