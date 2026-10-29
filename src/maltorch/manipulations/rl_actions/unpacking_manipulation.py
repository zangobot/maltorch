import os
import subprocess
import tempfile
import torch
from maltorch.manipulations.byte_manipulation import ByteManipulation
from maltorch.utils.utils import convert_torch_exe_to_bytes

class UnpackingManipulation(ByteManipulation):
    """Unpack a UPX‑compressed PE.

    The operation writes the sample to a temporary file, calls ``upx -d`` and
    reads the unpacked bytes back.  If UPX is missing a ``RuntimeError`` is
    raised.
    """

    def initialize(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zero_delta = torch.zeros((0,), dtype=torch.uint8, device=samples.device)
        return samples, zero_delta

    def _apply_manipulation(self, x: torch.Tensor, delta: torch.Tensor):
        processed = []
        samples = [x] if x.ndim==1 else x
        for i in range(len(samples)):
            bytez = convert_torch_exe_to_bytes(samples[i])
            fd, tmp = tempfile.mkstemp(suffix=".exe")
            with os.fdopen(fd, "wb") as f:
                f.write(bytez)
            unpacked_path = tmp + "_unpacked"
            try:
                ret = subprocess.call(
                    ["upx", tmp, "-d", "-o", unpacked_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError as e:
                os.unlink(tmp)
                raise RuntimeError("upx not found") from e
            if ret == 0:
                with open(unpacked_path, "rb") as f:
                    new_bytes = f.read()
                os.unlink(tmp)
                os.unlink(unpacked_path)
            else:
                new_bytes = bytez
                os.unlink(tmp)
            processed.append(torch.tensor(list(new_bytes), dtype=torch.uint8, device=x.device))
        max_len = max(p.shape[0] for p in processed)
        padded = [torch.nn.functional.pad(p, (0, max_len - p.shape[0])) for p in processed]
        new_x = torch.stack(padded)
        return new_x, delta
