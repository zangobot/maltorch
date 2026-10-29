import lief
from maltorch.utils.utils import convert_torch_exe_to_bytes
import torch
from maltorch.manipulations.byte_manipulation import ByteManipulation

from maltorch.manipulations.rl_actions.rebuild_util import rebuild

class EndgameBreakOptionalHeaderChecksumManipulation(ByteManipulation):
    """Set the PE optional header checksum to zero."""

    def initialize(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zero_delta = torch.zeros((0,), dtype=torch.uint8, device=samples.device)
        return samples, zero_delta

    def _apply_manipulation(self, x: torch.Tensor, delta: torch.Tensor):
        processed = []
        samples = [x] if x.ndim==1 else x
        for i in range(len(samples)):
            bytez = convert_torch_exe_to_bytes(samples[i])
            binary = lief.PE.parse(bytez)
            if binary is None:
                return x, delta
            binary.optional_header.checksum = 0
            new_bytes = rebuild(binary)
            processed.append(torch.tensor(list(new_bytes), dtype=torch.uint8, device=x.device))
        max_len = max(p.shape[0] for p in processed)
        padded = [torch.nn.functional.pad(p, (0, max_len - p.shape[0])) for p in processed]
        new_x = torch.stack(padded)
        return new_x, delta
