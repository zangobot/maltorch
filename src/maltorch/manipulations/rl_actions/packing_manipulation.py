import os
import random
import subprocess
import tempfile
import torch
from maltorch.manipulations.byte_manipulation import ByteManipulation

class PackingManipulation(ByteManipulation):
    """Compress the PE with UPX.

    The operation writes a temporary file, calls ``upx`` and reads the result back.
    If UPX is missing a :class:`RuntimeError` is raised.  The manipulation
    does not persist any temporary files.
    """

    def initialize(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zero_delta = torch.zeros((0,), dtype=torch.uint8, device=samples.device)
        return samples, zero_delta

    def _apply_manipulation(self, x: torch.Tensor, delta: torch.Tensor):
        processed = []
        for i in range(x.shape[0]):
            bytez = x[i].cpu().numpy().tobytes()
            # Write to temp file
            fd, tmp = tempfile.mkstemp(suffix=".exe")
            with os.fdopen(fd, "wb") as f:
                f.write(bytez)

            options = ["--force", "--overlay=copy"]
            comp = random.randint(1, 9)
            options.append(f"-{comp}")
            options += [f"--compress-exports={random.randint(0, 1)}"]
            options += [f"--compress-icons={random.randint(0, 3)}"]
            options += [f"--compress-resources={random.randint(0, 1)}"]
            options += [f"--strip-relocs={random.randint(0, 1)}"]

            packed_path = tmp + "_packed"
            try:
                ret = subprocess.call(
                    ["upx"] + options + [tmp, "-o", packed_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError as e:
                os.unlink(tmp)
                raise RuntimeError("upx not found") from e
            if ret == 0:
                with open(packed_path, "rb") as f:
                    new_bytes = f.read()
                os.unlink(tmp)
                os.unlink(packed_path)
            else:
                # keep original if packing failed
                new_bytes = bytez
                os.unlink(tmp)
            processed.append(torch.tensor(list(new_bytes), dtype=torch.uint8, device=x.device))
        max_len = max(p.shape[0] for p in processed)
        padded = [torch.nn.functional.pad(p, (0, max_len - p.shape[0])) for p in processed]
        new_x = torch.stack(padded)
        return new_x, delta
