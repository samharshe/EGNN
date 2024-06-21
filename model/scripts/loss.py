import torch
from torch import Tensor
from typing import Callable

def F_loss(F: Tensor, F_hat: Tensor, loss_fn: Callable) -> Tensor:
        # Euclidean distance between the predicted and actual force vectors
        F_error = torch.sqrt(torch.sum(torch.square(F - F_hat), dim=1))
        F_loss = loss_fn(F_error, torch.zeros_like(F_error))
        return F_loss