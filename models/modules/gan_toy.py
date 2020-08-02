import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, n_hidden, hidden_size, output_size):
        super().__init__()
        layers = []
        for _ in range(n_hidden):
            layers += [nn.Linear(input_size, hidden_size), nn.LeakyReLU(0.2)]
            input_size = hidden_size
        layers += [nn.Linear(hidden_size, output_size)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, input_size, n_hidden, hidden_size, output_size):
        super().__init__()
        self.latent_dim = input_size
        self.out_dim = output_size
        self.mlp = MLP(input_size, n_hidden, hidden_size, output_size)

    def forward(self, z):
        return torch.tanh(self.mlp(z))


class Discriminator(nn.Module):
    def __init__(self, input_size, n_hidden, hidden_size, output_size):
        super().__init__()
        self.mlp = MLP(input_size, n_hidden, hidden_size, output_size)

    def forward(self, z):
        return torch.sigmoid(self.mlp(z))
