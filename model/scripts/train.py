import torch
from torch import Tensor

import os

from torch_geometric.nn import global_add_pool

from losses import CalcF_squared_loss, CalcF_absolute_loss

import wandb

def train(model, optimizer, scheduler, loss_fn, train_loader, val_loader, config):
    # setting up wandb
    os.environ['WANDB_NOTEBOOK_NAME'] = 'train.ipynb'
    wandb.login()

    # reproducibility
    torch.manual_seed(2002)

    # learning rates
    learning_rates = [config.base_learning_rate]

    # initialize wandb run
    wandb.init(
        project = "EGNN",
        config = config,
    )
    
    # val statistics
    val_mean_losses = []

    # training loop occurs num_epochs times
    for epoch in range(config.num_epochs):
        # TRAINING
        # track gradients
        model.train()
        
        # dummy variable to track loss every 100 batches
        i = 0
        
        # loop through loader
        for data in train_loader:
            # clear gradients
            optimizer.zero_grad()

            # target values
            E = data.energy
            F = data.force
            
            # predictions from the model
            E_hat, F_hat = model(data)
            
            # squared error for energy loss
            E_loss = (1 - rho) * loss_fn(torch.squeeze(E_hat), E)

            # a version of squared error for force loss
            F_loss = rho * CalcF_squared_loss(F_hat, F)
            
            # canonical loss
            loss = E_loss + F_loss
        
            # calculate gradients
            loss.backward()
            
            # update
            optimizer.step()
            
            # save loss every 100 goes
            if i%100 == 0:
                wandb.log({"train_losses": loss.item()})
                wandb.log({"E_train_losses": E_loss.item()})
                wandb.log({"F_train_losses": F_loss.item()})
                
                # save learning rate
                lr = optimizer.param_groups[0]['lr']
                wandb.log({"training_rates": lr})
            i+=1
        
        # VAL
        epoch_losses = []
        epoch_E_losses = []
        epoch_F_losses = []
        
        # do not track gradients
        model.eval()
        
        # loop through val loader
        for data in val_loader:
            # target values
            E = data.energy
            F = data.force
            
            # predictions from the model
            E_hat, F_hat = model(data)
            
            # squared error for energy loss
            E_loss = (1 - rho) * loss_fn(torch.squeeze(E_hat), E)
            
            # a version of squared error for force loss
            F_loss = rho * CalcF_squared_loss(F_hat, F)
            
            # canonical loss
            loss =  E_loss + F_loss
            
            # track F_loss, E_loss, canonical loss
            epoch_losses.append(loss.item())
            epoch_E_losses.append(E_loss.item())
            epoch_F_losses.append(F_loss.item())
        
        # calculate the mean losses from this epoch
        epoch_mean_loss = torch.mean(torch.tensor(epoch_losses)).item()
        epoch_mean_E_loss = torch.mean(torch.tensor(epoch_E_losses)).item()
        epoch_mean_F_loss = torch.mean(torch.tensor(epoch_F_losses)).item()
        
        # save the mean canonical loss from this epoch for comparison to that of other epochs to determine whether to save weights
        val_mean_losses.append(epoch_mean_loss)
        
        # log mean losses with wandb
        wandb.log({"epoch_mean_loss": epoch_mean_loss})
        wandb.log({"epoch_mean_E_loss": epoch_mean_E_loss})
        wandb.log({"epoch_mean_F_loss": epoch_mean_F_loss})
        
        # print out the results of the epoch
        print(f'EPOCH {epoch+1} OF {num_epochs} | VAL MEAN LOSS: {epoch_mean_loss}')
        
        # if this is our best val performance yet, save the weights
        if min(val_mean_losses) == epoch_mean_loss:
            torch.save(model, f'../weights/{config.name}.pth')
            
        scheduler.step(epoch_mean_loss)