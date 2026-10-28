import random
from pathlib import Path
import torch
import lief
import array
from maltorch.manipulations.byte_manipulation import ByteManipulation

BASE_DIR = Path(__file__).resolve().parent.parent

def _random_length():
    return 2 ** random.randint(5, 8)

# Rebuild helper

def _rebuild(binary, imports=False):
    config = lief.PE.Builder.config_t()
    config.imports = imports
    return binary.write_to_bytes(config)

class EndgameSectionInjectionManipulation(ByteManipulation):
    """Add a new random section to each PE sample."""

    def initialize(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zero_delta = torch.zeros((0,), dtype=torch.uint8, device=samples.device)
        return samples, zero_delta

    def _apply_manipulation(self, x: torch.Tensor, delta: torch.Tensor):
        processed = []
        for i in range(x.shape[0]):
            bytez = x[i].cpu().numpy().tobytes()
            binary = lief.PE.parse(bytez)
            random.seed()  # per sample randomness
            new_section = lief.PE.Section(
                "".join(chr(random.randrange(ord('.'), ord('z'))) for _ in range(6)))
            upper = random.randrange(256)
            L = _random_length()
            new_section.content = [random.randint(0, upper) for _ in range(L)]
            new_section.virtual_address = max(
                [s.virtual_address + s.size for s in binary.sections])
            binary.add_section(
                new_section,
                random.choice([
                    lief.PE.SECTION_TYPES.BSS,
                    lief.PE.SECTION_TYPES.DATA,
                    lief.PE.SECTION_TYPES.EXPORT,
                    lief.PE.SECTION_TYPES.IDATA,
                    lief.PE.SECTION_TYPES.RELOCATION,
                    lief.PE.SECTION_TYPES.RESOURCE,
                    lief.PE.SECTION_TYPES.TEXT,
                    lief.PE.SECTION_TYPES.TLS_,
                    lief.PE.SECTION_TYPES.UNKNOWN,
                ]),
            )
            new_bytes = _rebuild(binary)
            processed.append(torch.tensor(list(new_bytes), dtype=torch.uint8, device=x.device))
        max_len = max(p.shape[0] for p in processed)
        padded = [torch.nn.functional.pad(p, (0, max_len - p.shape[0])) for p in processed]
        new_x = torch.stack(padded)
        return new_x, delta
