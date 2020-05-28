import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.pytorch_utils as ptu
from src.sn import SpectralNorm 



class MLP_d(nn.Module):
    def __init__(self, input_size, n_hidden, hidden_size, output_size):
        super().__init__()
        layers = []
        for _ in range(n_hidden):
            layers.append(SpectralNorm(nn.Linear(input_size, hidden_size)))
            layers.append(nn.ELU(0.2))
            input_size = hidden_size
        layers.append(SpectralNorm(nn.Linear(hidden_size, output_size)))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class MLP_g(nn.Module):
    def __init__(self, input_size, n_hidden, hidden_size, output_size):
        super().__init__()
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.LeakyReLU(0.2))
            input_size = hidden_size
        layers.append(SpectralNorm(nn.Linear(hidden_size, output_size)))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, n_hidden, hidden_size, data_dim):
        super().__init__()
        layers = []
        self.latent_dim = latent_dim
        self.out_dim = data_dim
        self.mlp = MLP_g(latent_dim, n_hidden, hidden_size, data_dim)
    
    def forward(self, z):
        return torch.tanh(self.mlp(z))
        
    def sample(self, n):
        z = ptu.normal(ptu.zeros(n, self.latent_dim), ptu.ones(n, self.latent_dim))
        return self.forward(z)

class Discriminator(nn.Module):
    def __init__(self, latent_dim, n_hidden, hidden_size, data_dim):
        super().__init__()
        self.mlp = MLP_d(latent_dim, n_hidden, hidden_size, data_dim)
    
    def forward(self, z):
        return torch.sigmoid(self.mlp(z)) 
