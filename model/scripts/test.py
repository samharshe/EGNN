import torch
from torch import Tensor

import os

from torch_geometric.nn import global_add_pool

from loss import F_loss

import wandb

def test(model, loss_fn, test_loader, config):
    # do not track gradients
    model.eval()
    
    # to log everything
    wandb.login()
    
    # for concision
    rho = config["rho"]

    # test statistics using the same loss function as training
    test_squared_losses = []
    test_E_squared_losses = []
    test_F_squared_losses = []

    # test statistics using MAE for comparison with other benchmarks
    test_absolute_losses = []
    test_E_absolute_losses = []
    test_F_absolute_losses = []
            
    for data in test_loader:
        # target values
        E = data.energy
        F = data.force
        
        # predictions from the model
        E_hat, F_hat = model(data)
        torch.squeeze_(E_hat)
        
        # squared error for energy loss
        E_squared_loss = (1-rho) * loss_fn(E_hat, E)
        
        # a version of squared error for force loss
        F_squared_loss = rho * F_loss(F_hat, F, loss_fn)
        
        # canonical loss
        squared_loss = E_squared_loss + F_squared_loss
        
        # squared error for energy loss
        E_absolute_loss = (1 - rho) * torch.mean(torch.abs(E_hat-E))
        
        # a version of squared error for force loss
        F_absolute_loss = rho * F_loss(F_hat, F, torch.mean)
        
        # canonical loss
        absolute_loss = E_absolute_loss + F_absolute_loss
        
        # save squared losses
        test_squared_losses.append(squared_loss.item())
        test_E_squared_losses.append(E_squared_loss.item())
        test_F_squared_losses.append(F_squared_loss.item())
        
        # save absolute losses
        test_absolute_losses.append(absolute_loss.item())
        test_E_absolute_losses.append(E_absolute_loss.item())
        test_F_absolute_losses.append(F_absolute_loss.item())

    # calculate and log mean test losses
    test_mean_squared_loss = torch.mean(torch.tensor(test_squared_losses)).item()
    test_mean_E_squared_loss = torch.mean(torch.tensor(test_E_squared_losses)).item()
    test_mean_F_squared_loss = torch.mean(torch.tensor(test_F_squared_losses)).item()

    wandb.log({"test_mean_squared_loss": test_mean_squared_loss})
    wandb.log({"test_mean_E_squared_loss": test_mean_E_squared_loss})
    wandb.log({"test_mean_F_squared_loss": test_mean_F_squared_loss})

    test_mean_absolute_loss = torch.mean(torch.tensor(test_absolute_losses)).item()
    test_mean_E_absolute_loss = torch.mean(torch.tensor(test_E_absolute_losses)).item()
    test_mean_F_absolute_loss = torch.mean(torch.tensor(test_F_absolute_losses)).item()

    wandb.log({"test_mean_absolute_loss": test_mean_absolute_loss})
    wandb.log({"test_mean_E_absolute_loss": test_mean_E_absolute_loss})
    wandb.log({"test_mean_F_absolute_loss": test_mean_F_absolute_loss})

    # print mean test losses
    print(f'TEST MEAN SQUARED LOSS: {test_mean_squared_loss}')
    print(f'TEST MEAN ABSOLUTE LOSS: {test_mean_squared_loss}')

    # end wandb run
    wandb.finish()