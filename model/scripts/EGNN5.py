import torch
from torch import Tensor

from torch.nn import Embedding, Linear, SiLU
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import MessagePassing

from gaussian_rbf import gaussian_rbf

class EGNN5(MessagePassing):
    def __init__(self):
        super().__init__()
        
        # activation function
        self.act = SiLU()
        
        # initialize layers
        # 118 atomic numbers into 32-dimensional space
        self.embedding = Embedding(118,64)
        
        # 64 dimensions for the embedding of the neighbor
        # 8 for the embedding of the distance
        self.message_lin = Linear(64 + 8, 64)
        
        # 64 dimensions for the current node embedding
        # 64 for the message
        self.update_lin = Linear(64 + 64, 64)
        
        # 64 dimensions for the embedding in and out
        self.atomwise_lin1 = Linear(64, 64)
        self.atomwise_lin2 = Linear(64, 64)
        self.atomwise_lin3 = Linear(64, 64)
        
        # compress the 32-dimensional node embedding to 1 dimension
        self.compress_lin1 = Linear(64, 8)
        self.compress_lin2 = Linear(8, 1)
        
    def forward(self, data):
        # get attributes out of data object
        edge_index = data.edge_index
        z = data.z
        pos = data.pos
        
        # force is the negative gradient of energy with respect to position, so pos must be on the computational graph
        pos.requires_grad_(True)
        
        # calculate edge distances and turn them into a vector through Gaussian RBF
        idx1, idx2 = edge_index
        edge_attr = torch.norm(pos[idx1] - pos[idx2], p=2, dim=-1).view(-1, 1)
        gaussian_edge_attr = gaussian_rbf(edge_attr)
        
        # embed
        E_hat = self.embedding(z)
        E_hat = self.act(E_hat)
        
        # message passing x 3
        # message passing 1
        E_hat = self.propagate(edge_index, x=E_hat, edge_attr=gaussian_edge_attr)
        E_hat = self.act(E_hat)
        E_hat = self.atomwise_lin1(E_hat)
        E_hat = self.act(E_hat)
        
        # message passing 2
        E_hat = self.propagate(edge_index, x=E_hat, edge_attr=gaussian_edge_attr)
        E_hat = self.act(E_hat)
        E_hat = self.atomwise_lin1(E_hat)
        E_hat = self.act(E_hat)
        
        # message passing 3
        E_hat = self.propagate(edge_index, x=E_hat, edge_attr=gaussian_edge_attr)
        E_hat = self.act(E_hat)
        E_hat = self.atomwise_lin1(E_hat)
        E_hat = self.act(E_hat)

        # compression
        E_hat = self.compress_lin1(E_hat)
        E_hat = self.act(E_hat)
        E_hat = self.compress_lin2(E_hat)
        E_hat = self.act(E_hat)
        E_hat = global_add_pool(E_hat, data.batch)
        
        # calculate the energy prediction as the negative gradient of energy with respect to position, retaining the computational graph for backprop
        F_hat = -torch.autograd.grad(E_hat.sum(), pos, retain_graph=True)[0]
        
        # return a tuple of the predictions
        return E_hat, F_hat
    
    def message(self, x_j, edge_attr):
        # concatenate the vectors
        lin_in = torch.cat((x_j, edge_attr), dim=1).float()
        
        # pass them into the linear layer
        out = self.message_lin(lin_in)
        
        # return the output
        return out
    
    def update(self, aggr_out, x):
        # concatenate the vectors
        lin_in = torch.cat((aggr_out, x), dim=1).float()
        
        # pass them into the linear layer
        out = self.update_lin(lin_in)
        
        # return the output
        return out