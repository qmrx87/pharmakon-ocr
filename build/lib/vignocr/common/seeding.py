"""Deterministic seeding across stdlib, NumPy, and (lazily) PyTorch.

Call ``seed_everything(seed)`` at the top of every train/eval/infer entrypoint.
PyTorch is imported lazily so the core (CPU) stack never requires the [ml] extra.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int = 1337, *, deterministic_torch: bool = True) -> int:
    """Seed all RNGs. Returns the seed (handy for logging into the run dir)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # numpy is a core dep, but never let seeding crash a run
        pass

    try:
        import torch  # lazy: only present with the [ml] extra

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    return seed
