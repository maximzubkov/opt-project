import torch
import torch.nn as nn

from src import SpectralNorm


class MLP_d(nn.Module):
    def __init__(self, input_size, n_hidden, hidden_size, output_size):
        super().__init__()
        layers = []
        for _ in range(n_hidden):
            layers += [SpectralNorm(nn.Linear(input_size, hidden_size)), nn.ELU(0.2)]
            input_size = hidden_size
        layers += [SpectralNorm(nn.Linear(hidden_size, output_size))]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLP_g(nn.Module):
    def __init__(self, input_size, n_hidden, hidden_size, output_size):
        super().__init__()
        layers = []
        for _ in range(n_hidden):
            layers += [nn.Linear(input_size, hidden_size), nn.LeakyReLU(0.2)]
            input_size = hidden_size
        layers += [SpectralNorm(nn.Linear(hidden_size, output_size))]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SN_Generator(nn.Module):
    def __init__(self, input_size, n_hidden, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mlp = MLP_g(input_size, n_hidden, hidden_size, output_size)

    def forward(self, z):
        return torch.tanh(self.mlp(z))


class SN_Discriminator(nn.Module):
    def __init__(self, input_size, n_hidden, hidden_size, output_size):
        super().__init__()
        self.mlp = MLP_d(input_size, n_hidden, hidden_size, output_size)

    def forward(self, z):
        return torch.sigmoid(self.mlp(z))
