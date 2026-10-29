import random
from pathlib import Path
import torch
import lief
import array
from maltorch.utils.utils import convert_torch_exe_to_bytes
from maltorch.manipulations.byte_manipulation import ByteManipulation

# Ensure section names file exists
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
SECTION_NAMES_PATH = BASE_DIR / "gym_malware" / "envs" / "controls" / "section_names.txt"
if not SECTION_NAMES_PATH.exists():
    import urllib.request
    url = "https://raw.githubusercontent.com/endgameinc/gym-malware/master/gym_malware/envs/controls/section_names.txt"
    urllib.request.urlretrieve(url, str(SECTION_NAMES_PATH))

with open(SECTION_NAMES_PATH) as f:
    COMMON_SECTION_NAMES = [line.strip() for line in f if line.strip()]


def _rebuild(binary, imports=False):
    config = lief.PE.Builder.config_t()
    config.imports = imports
    return binary.write_to_bytes(config)

class EndgameSectionRenameManipulation(ByteManipulation):
    """Randomly rename one section in the PE header for each sample."""

    def initialize(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zero_delta = torch.zeros((0,), dtype=torch.uint8, device=samples.device)
        return samples, zero_delta

    def _apply_manipulation(self, x: torch.Tensor, delta: torch.Tensor):
        processed = []
        samples = [x] if x.ndim == 1 else x
        for i in range(len(samples)):
            bytez = convert_torch_exe_to_bytes(samples[i])
            binary = lief.PE.parse(bytez)
            if binary is None:
                continue
            random.seed()  # ensure randomness per sample
            targeted_section = random.choice(binary.sections)
            targeted_section.name = random.choice(COMMON_SECTION_NAMES)[:7]
            new_bytes = _rebuild(binary)
            processed.append(torch.tensor(list(new_bytes), dtype=torch.uint8, device=x.device))
        if not processed:
            return x, delta
        max_len = max(p.shape[0] for p in processed)
        padded = [torch.nn.functional.pad(p, (0, max_len - p.shape[0])) for p in processed]
        new_x = torch.stack(padded)
        return new_x, delta
