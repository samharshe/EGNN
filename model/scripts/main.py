import torch
from torch import Tensor

import os

from torch.nn import MSELoss
from torch_geometric.nn import global_add_pool

from losses import CalcF_squared_loss, CalcF_absolute_loss
from EGNN5 import EGNN5

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb

# setting up wandb
os.environ['WANDB_NOTEBOOK_NAME'] = 'EGNN4.ipynb'
wandb.login()

# reproducibility
torch.manual_seed(2002)

# hyperparameters saved to config dict
config = {
    'base_learning_rate': 0.001,
    'num_epochs': 50,
    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_mode': 'min',
    'scheduler_factor': 0.32, 
    'scheduler_patience': 1,
    'scheduler_threshold': 0,
    'training_loss_fn': 'MSELoss',
    'rho': 1-1e-1,
    'batch_size': 32
}

# initialize the star of the show
model = EGNN5()

# I couldn't think of a concise way to initialize optimizer, scheduler, and loss_fn based on the contents of config
# this is all for show anyway, but it would be nice to have a natural way of doing this that generalizes when I am selecting hyperparameters more carefully
optimizer = Adam(model.parameters(), lr=config.base_learning_rate)
scheduler = ReduceLROnPlateau(
    optimizer=optimizer, 
    mode=config.scheduler_mode, 
    factor=config.scheduler_factor, 
    patience=config.scheduler_patience, 
    threshold=config.scheduler_threshold
    )
loss_fn = MSELoss()

# get dataloaders
train_loader, val_loader, test_loader = get_MD17_data_loaders(train_split=0.8, val_split=0.1, test_split = 0.1, batch_size=config.batch_size)

# initialize wandb run
wandb.init(
    project = "EGNN",
    config = config,
)

train(model, optimizer, scheduler, loss_fn, train_loader, val_loader, config)
test(model, loss_fn, test_loader, config)