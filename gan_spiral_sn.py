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
        layers.append(nn.Linear(hidden_size, output_size))
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
        return self.mlp(z)
        
    def sample(self, n):
        z = ptu.normal(ptu.zeros(n, self.latent_dim), ptu.ones(n, self.latent_dim))
        return self.forward(z)

class Discriminator(nn.Module):
    def __init__(self, latent_dim, n_hidden, hidden_size, data_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.mlp = MLP_d(latent_dim, n_hidden, hidden_size, data_dim)
    
    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        print(out.shape)
        return torch.sigmoid(self.mlp(out)) 
