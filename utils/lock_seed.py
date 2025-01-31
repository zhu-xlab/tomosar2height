import numpy as np
import random
import torch


def lock_seed(seed: int = 0):
    """
    Lock the random seed for reproducible results.

    Args:
        seed (int): The seed value to set for reproducibility.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
