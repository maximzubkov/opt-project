import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.pytorch_utils as ptu
from src.sn import SpectralNorm 
import numpy as np
from torch.autograd import Variable


opt = {}
opt["img_size"] = 16
opt["latent_dim"] = 32
opt["channels"] = 1
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt["img_size"] // 4
        self.l1 = nn.Sequential(nn.Linear(opt["latent_dim"], 32 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, opt["channels"], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 32, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
    
    def sample(self, n):
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (n, opt["latent_dim"]))).to(ptu.device))
        out = self.l1(z)
        out = out.view(out.shape[0], 32, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
        


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt["channels"], 16, bn=False),
            *discriminator_block(16, 32),
        )

        # The height and width of downsampled image
        ds_size = opt["img_size"] // 2 ** 2
        self.adv_layer = nn.Linear(32 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity