import json
import random
from pathlib import Path
import torch
import lief
from maltorch.manipulations.byte_manipulation import ByteManipulation
from maltorch.manipulations.rl_actions.rebuild_util import rebuild
from maltorch.utils.utils import convert_torch_exe_to_bytes

# Load the common imports mapping once at module load
BASE_DIR = Path(__file__).resolve().parent.parent  # repository root
JSON_PATH = BASE_DIR / "gym_malware" / "envs" / "controls" / "small_dll_imports.json"
# If the file is not present, download it from GitHub
# Ensure directory exists before downloading
if not JSON_PATH.exists():
    import urllib.request
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/endgameinc/gym-malware/master/gym_malware/envs/controls/small_dll_imports.json"
    urllib.request.urlretrieve(url, str(JSON_PATH))

with open(JSON_PATH) as f:
    COMMON_IMPORTS = json.load(f)

# Helper function to add an import (copied from gym-malware)

def _add_imports(bytez: bytes, seed=None) -> bytes:
    random.seed(seed)
    binary = lief.PE.parse(bytez)
    if binary is None:
        return None
    libname = random.choice(list(COMMON_IMPORTS.keys()))
    funcname = random.choice(list(COMMON_IMPORTS[libname]))
    lowerlibname = libname.lower()
    lib = None
    for im in binary.imports:
        if im.name.lower() == lowerlibname:
            lib = im
            break
    if lib is None:
        lib = binary.add_import(libname)
    names = set([e.name for e in lib.entries])
    if funcname not in names:
        lib.add_entry(funcname)
    binary = rebuild(binary, imports=True)
    return bytez

# Re‑use the builder from gym-malware for consistency



# Re‑use the builder from gym-malware for consistency

class EndgameImportsInjectionManipulation(ByteManipulation):
    """Inject a random unused import into each sample.

    The manipulation appends an additional library/function entry to the PE
    imports table.  The operation is performed on each sample independently.
    """

    def initialize(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # No delta needed – return an empty tensor
        zero_delta = torch.zeros((0,), dtype=torch.uint8, device=samples.device)
        return samples, zero_delta

    def _apply_manipulation(self, x: torch.Tensor, delta: torch.Tensor):
        # Process each sample in the batch
        processed = []
        samples = [x] if x.ndim==1 else x
        for i in range(len(samples)):
            bytez = convert_torch_exe_to_bytes(samples[i])
            modified = _add_imports(bytez)
            if modified is not None:
                processed.append(torch.tensor(list(modified), dtype=torch.uint8, device=x.device))
        # Pad to the longest length
        if not processed:
            return x, delta
        max_len = max(p.shape[0] for p in processed)
        padded = [torch.nn.functional.pad(p, (0, max_len - p.shape[0])) for p in processed]
        new_x = torch.stack(padded)
        return new_x, delta
