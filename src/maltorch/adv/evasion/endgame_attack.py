from typing import Union, List, Callable

from maltorch.adv.evasion.rl_based_attack import ReinforcementLearningBackendAttack
from maltorch.manipulations.sequential_manipulation import SequentialManipulation
from maltorch.manipulations.rl_actions.endgame_break_optional_header_checksum_manipulation import EndgameBreakOptionalHeaderChecksumManipulation
from maltorch.manipulations.rl_actions.endgame_entry_point_manipulation import EndgameEntryPointManipulation
from maltorch.manipulations.rl_actions.endgame_import_inj_manipulation import EndgameImportsInjectionManipulation
from maltorch.manipulations.rl_actions.endgame_padding_manipulation import EndgamePaddingManipulation
from maltorch.manipulations.rl_actions.endgame_remove_debug_manipulation import EndgameRemoveDebugManipulation
from maltorch.manipulations.rl_actions.endgame_remove_signature_manipulation import EndgameRemoveSignatureManipulation
from maltorch.manipulations.rl_actions.endgame_section_inj_manipulation import EndgameSectionInjectionManipulation
from maltorch.manipulations.rl_actions.endgame_section_rename_manipulations import EndgameSectionRenameManipulation
from maltorch.manipulations.rl_actions.packing_manipulation import PackingManipulation
from maltorch.manipulations.rl_actions.unpacking_manipulation import UnpackingManipulation
from secmlt.trackers.trackers import Tracker

from maltorch.adv.evasion.base_optim_attack_creator import (
    BaseOptimAttackCreator,
    OptimizerBackends,
)
from maltorch.optim.halting import Halting
from maltorch.optim.optimizer_factory import MalwareOptimizerFactory



class EndgameAttack(BaseOptimAttackCreator):
    @staticmethod
    def get_backends() -> set[str]:
        return {OptimizerBackends.RL}

    @staticmethod
    def _get_nevergrad_implementation():
        raise NotImplementedError()

    @staticmethod
    def _get_native_implementation() :
        raise NotImplementedError()

    def __new__(
            cls,
            query_budget: int,
            training_budget = 1000,
            training_fraction = 0.5,
            lr=1e-5,
            y_target: Union[int, None] = None,
            random_init: bool = False,
            device: str = "cpu",
            model_outputs_logits: bool = True,
            trackers: Union[List[Tracker], Tracker] = None,
            backend: str = OptimizerBackends.RL,
            early_stopping: Halting = None
    ) -> Callable:
        policy_instance = MalwareOptimizerFactory.create_dqn_policy(learning_rate=lr)
        manipulations = SequentialManipulation([
            EndgameBreakOptionalHeaderChecksumManipulation,
            EndgameEntryPointManipulation,
            EndgameImportsInjectionManipulation,
            EndgamePaddingManipulation,
            EndgameRemoveDebugManipulation,
            EndgameRemoveSignatureManipulation,
            EndgameSectionInjectionManipulation,
            EndgameSectionRenameManipulation,
            PackingManipulation,
            UnpackingManipulation
        ])
        return ReinforcementLearningBackendAttack(
            optimizer_cls=policy_instance,
            manipulation_function=manipulations,
            training_budget = training_budget,
            training_fraction = training_fraction,
            query_budget=query_budget,
            device=device,
        )
