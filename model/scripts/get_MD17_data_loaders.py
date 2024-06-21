import torch
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

def get_MD17_data_loaders(train_split, val_split, test_split, batch_size):
    # make sure we are doing everything with the whole dataset
    s = train_split + val_split + test_split
    assert s == 1, f"train_split, val_split, and test_split must sum to 1. got: {s}"
    
    # load in the dataset
    dataset = MD17(root='../../data/EGNN2/benzene', name='benzene', pre_transform=None, transform=None)

    # split defined by the argument of the function
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # for reproducibility
    torch.manual_seed(2002)
    
    # build train, val, test datasets out of main dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # turn into DataLoaders for batching efficiency
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # return the DataLoaders
    return train_loader, val_loader, test_loader