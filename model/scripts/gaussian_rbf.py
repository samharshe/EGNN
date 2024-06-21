import torch
from torch import Tensor

def gaussian_rbf(x: Tensor) -> Tensor:
    sigma, c_min, c_max, c_inc, num_c = 0.05, 0, 1.6, 8
    c_inc = c_max / num_c
    
    cs = torch.arange(c_min, c_max, c_inc)
    
    return torch.exp(-torch.square((x - cs)) / (2*sigma**2)).float()