# PyTorch
import torch

# smarter force losses
from loss import F_loss_fn
from torch.nn import L1Loss

# logging results
import wandb

# type annotations
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.loader import DataLoader
from typing import Callable, Dict

def test_model(model: MessagePassing, loss_fn: Callable, test_loader: DataLoader, rho: float) -> None:
    # do not track gradients
    model.eval()
    
    # test statistics using the same loss function as training
    test_losses = []
    test_E_losses = []
    test_F_losses = []

    # test statistics using MAE for comparison with other benchmarks
    test_absolute_losses = []
    test_E_absolute_losses = []
    test_F_absolute_losses = []
    
    # iterate through test_loader        
    for data in test_loader:
        # target values
        E = data.energy
        F = data.force
        
        # predictions from the model
        E_hat, F_hat = model(data)
        E_hat.squeeze_(dim=1)
        
        # squared error for energy loss
        E_loss = (1-rho) * loss_fn(E_hat, E)
        
        # a version of squared error for force loss
        F_loss = rho * F_loss_fn(F_hat, F, loss_fn)
        
        # canonical loss
        loss = E_loss + F_loss
        
        # absolute error 
        absolute_loss_fn = L1Loss()
        
        # absolute error for energy loss
        E_absolute_loss = (1 - rho) * absolute_loss_fn(E_hat, E)
        
        # a version of absolute error for force loss
        F_absolute_loss = rho * F_loss_fn(F_hat, F, absolute_loss_fn)
        
        # absolute loss
        absolute_loss = E_absolute_loss + F_absolute_loss
        
        # save squared losses
        test_losses.append(loss.item())
        test_E_losses.append(E_loss.item())
        test_F_losses.append(F_loss.item())
        
        # save absolute losses
        test_absolute_losses.append(absolute_loss.item())
        test_E_absolute_losses.append(E_absolute_loss.item())
        test_F_absolute_losses.append(F_absolute_loss.item())
        
    # calculate and log mean test losses
    test_mean_loss = torch.mean(torch.tensor(test_lossses)).item()
    test_mean_E_loss = torch.mean(torch.tensor(test_E_losses)).item()
    test_mean_F_loss = torch.mean(torch.tensor(test_F_losses)).item()

    wandb.log({"test_mean_loss": test_mean_loss})
    wandb.log({"test_mean_E_loss": test_mean_E_loss})
    wandb.log({"test_mean_F_loss": test_mean_F_loss})

    test_mean_absolute_loss = torch.mean(torch.tensor(test_absolute_losses)).item()
    test_mean_E_absolute_loss = torch.mean(torch.tensor(test_E_absolute_losses)).item()
    test_mean_F_absolute_loss = torch.mean(torch.tensor(test_F_absolute_losses)).item()

    wandb.log({"test_mean_absolute_loss": test_mean_absolute_loss})
    wandb.log({"test_mean_E_absolute_loss": test_mean_E_absolute_loss})
    wandb.log({"test_mean_F_absolute_loss": test_mean_F_absolute_loss})
    
    # print mean test losses
    print(f'TEST MEAN LOSS: {test_mean_loss}')
    print(f'TEST MEAN ABSOLUTE LOSS: {test_mean_absolute_loss}')

    # end wandb run
    wandb.finish()