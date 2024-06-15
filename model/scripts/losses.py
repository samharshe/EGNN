import torch
from torch import Tensor

def CalcF_squared_loss(F: Tensor, F_hat: Tensor) -> Tensor:
        # average square of the magnitude of the difference between the predicted and actual force vectors on each atom
        F_error = F_hat - F
        F_squared_error = torch.square(F_error)
        F_loss = torch.div(torch.sum(F_squared_error), F.size()[0])
        return F_loss
    
def CalcF_absolute_loss(F: Tensor, F_hat: Tensor) -> Tensor:
        # average of the absolute value of the difference between the predicted and actual force vectors on each atom
        F_error = torch.abs(F_hat - F)
        F_loss = torch.div(torch.sum(F_error), F.size()[0])
        return F_loss