import torch
from torch import Tensor
import numpy as np

def gaussian_rbf(x: Tensor) -> Tensor:
    cs = torch.tensor(np.arange(0,1.6,0.2))
    return torch.exp(torch.square((x - cs)) / -.005).float()