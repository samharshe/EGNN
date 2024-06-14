import torch
from torch import Tensor

def gaussian_rbf(c: Tensor, x: Tensor, sigma: float = float(1)) -> Tensor:
    return torch.exp(torch.mul(torch.div(torch.pow(torch.dist(c,x),2), 2 * sigma ** 2),-1))