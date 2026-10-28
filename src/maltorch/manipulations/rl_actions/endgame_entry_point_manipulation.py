import random
import struct
import lief
import torch
from maltorch.manipulations.byte_manipulation import ByteManipulation

from maltorch.manipulations.rl_actions.rebuild_util import rebuild

class EndgameEntryPointManipulation(ByteManipulation):
    """Create a new section that jumps to the original entry point and
    redirect the PE's entry point to this new section.
    """

    def initialize(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zero_delta = torch.zeros((0,), dtype=torch.uint8, device=samples.device)
        return samples, zero_delta

    def _apply_manipulation(self, x: torch.Tensor, delta: torch.Tensor):
        processed = []
        for i in range(x.shape[0]):
            bytez = x[i].cpu().numpy().tobytes()
            binary = lief.PE.parse(bytez)

            random.seed()  # per sample randomness
            entry_point = binary.optional_header.addressof_entrypoint
            entryname = binary.section_from_rva(entry_point).name

            new_section = lief.PE.Section(
                entryname + "".join(chr(random.randrange(ord('.'), ord('z'))) for _ in range(3)))
            # push [old_entry_point]; ret
            new_section.content = [0x68] + list(struct.pack("<I", entry_point + 0x10000)) + [0xc3]
            new_section.virtual_address = max(
                [s.virtual_address + s.size for s in binary.sections])
            SECTION_TYPES.TEXT
            new_section.characteristics = ...
            binary.add_section(new_section)

            binary.optional_header.addressof_entrypoint = new_section.virtual_address

            new_bytes = rebuild(binary)
            processed.append(torch.tensor(list(new_bytes), dtype=torch.uint8, device=x.device))
        max_len = max(p.shape[0] for p in processed)
        padded = [torch.nn.functional.pad(p, (0, max_len - p.shape[0])) for p in processed]
        new_x = torch.stack(padded)
        return new_x, delta
