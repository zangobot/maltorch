import random
import string
from pathlib import Path
from typing import Union

import lief.PE
import torch

from maltorch.initializers.initializers import IdentityInitializer
from maltorch.manipulations.byte_manipulation import ByteManipulation
from maltorch.utils.utils import convert_torch_exe_to_list
from maltorch.utils.zangope import Binary


class FastGAMMASectionInjectionManipulation(ByteManipulation):
    def __init__(
            self,
            benignware_folder: Union[Path, list[str], list[Path]],
            which_sections=None,
            how_many_sections: int = 75,
            domain_constraints=None,
            perturbation_constraints=None,
    ):
        self.how_many_sections = how_many_sections
        if which_sections is None:
            which_sections = [".rdata"]
        self.benignware_folder = benignware_folder
        self.which_sections = which_sections
        if domain_constraints is None:
            domain_constraints = []
        if perturbation_constraints is None:
            perturbation_constraints = []
        self._sections = []
        self._names = [
            ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)) for _ in range(self.how_many_sections)
        ]
        for path in sorted(self.benignware_folder.glob("*")):
            if not lief.is_pe(str(path)):
                continue
            lief_pe = lief.parse(str(path))
            for s in lief_pe.sections:
                if s.name not in self.which_sections:
                    continue
                if len(self._sections) < self.how_many_sections:
                    self._sections.append(list(s.content))
        super().__init__(IdentityInitializer(), domain_constraints, perturbation_constraints)

    def _apply_manipulation(
            self, x: torch.Tensor, delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pe = Binary(bytez=bytearray(convert_torch_exe_to_list(x)))
        for delta_i, content, name in zip(delta.squeeze(), self._sections, self._names):
            pe.add_robust_section(name, 0x40000040, bytearray(content[:int(len(content) * delta_i)]))
        x = torch.atleast_2d(torch.Tensor(pe.get_bytes()).long()).to(x.device)
        return x, delta

    def initialize(self, samples: torch.Tensor):
        delta = torch.zeros((samples.shape[0], self.how_many_sections))
        if self.initializer.random_init:
            delta = torch.rand((samples.shape[0], self.how_many_sections))
        return samples, delta