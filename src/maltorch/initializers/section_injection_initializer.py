import torch

from maltorch.initializers.initializers import ByteBasedInitializer
from maltorch.utils.pe_operations import section_injection_manipulation


class SectionInjectionInitializer(ByteBasedInitializer):
    def __init__(
        self,
        how_many_sections: int,
        size_per_section: int = 0x200,
        random_init: bool = False,
    ):
        super().__init__(random_init=random_init)
        self.how_many_sections = how_many_sections
        self.size_per_section = size_per_section

    def __call__(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        X = []
        deltas = []
        indexes = []
        for x_i in x:
            x_i, shift_indexes = section_injection_manipulation(
                x_i, self.how_many_sections, self.size_per_section
            )
            X.append(x_i)
            delta = (
                torch.zeros(len(shift_indexes))
                if not self.random_init
                else torch.randint(0, 255, (len(shift_indexes),))
            )
            deltas.append(delta)
            indexes.append(shift_indexes)
        x, delta, indexes = self._pad_samples_same_length(X, deltas, indexes)
        return x, delta, indexes
