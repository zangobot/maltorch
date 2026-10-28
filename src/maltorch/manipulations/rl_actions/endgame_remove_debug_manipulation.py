import lief
import torch
from maltorch.manipulations.byte_manipulation import ByteManipulation
from maltorch.manipulations.rl_actions.rebuild_util import rebuild


class EndgameRemoveDebugManipulation(ByteManipulation):
    """Remove the PE debug directory if present."""

    def initialize(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zero_delta = torch.zeros((0,), dtype=torch.uint8, device=samples.device)
        return samples, zero_delta

    def _apply_manipulation(self, x: torch.Tensor, delta: torch.Tensor):
        processed = []
        for i in range(x.shape[0]):
            bytez = x[i].cpu().numpy().tobytes()
            binary = lief.PE.parse(bytez)

            if binary.has_debug:
                for e in binary.data_directories:
                    if e.type == lief.PE.DATA_DIRECTORY.DEBUG:
                        e.rva = 0
                        e.size = 0
                        break
                new_bytes = rebuild(binary)
            else:
                new_bytes = bytez
            processed.append(torch.tensor(list(new_bytes), dtype=torch.uint8, device=x.device))
        max_len = max(p.shape[0] for p in processed)
        padded = [torch.nn.functional.pad(p, (0, max_len - p.shape[0])) for p in processed]
        new_x = torch.stack(padded)
        return new_x, delta
