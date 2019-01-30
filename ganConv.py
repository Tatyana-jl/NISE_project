import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils


class Generator(nn.Module):

    def __init__(self, parameters):
        super(Generator, self).__init__()

        self.label_emb_gen = nn.Embedding(parameters['n_classes'], parameters['n_classes'])
        self.img_shape = [parameters['img_size'], parameters['img_size']]

        self.dense = nn.Sequential(
            nn.Linear(parameters['latent_dim']+parameters['n_classes'], 128*parameters['img_size']**2),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        def block_conv(in_chan, out_chan, filter):
            layers = [nn.Conv2d(in_chan, out_chan, filter, stride=1, padding=1)]
            layers.append(nn.BatchNorm2d(out_chan, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            *block_conv(128, 128, 3),
            nn.Upsample(scale_factor=2),
            *block_conv(128, 64, 3),
            nn.Conv2d(64, parameter['channels'], 3),
            nn.Tanh()
        )

        def forward(self, noise, labels):
            # Concatenate label embedding and image to produce input
            gen_input = torch.cat((self.label_emb_gen(labels), noise), -1)
            to_conv = self.dense(gen_input)
            to_conv = to_conv.view(to_conv.shape[0], 128, parameters['img_size'], parameters['image_size'])
            img = self.conv(to_conv)
            img = img.view(img.size(0), *self.img_shape)
            return img


class Discriminator(nn.Module):
    def __init__(self, parameters):
        super(Discriminator, self).__init__()

        self.label_emb_dis = nn.Embedding(parameters['n_classes'], parameters['n_classes'])
        self.img_shape = [parameters['img_size'], parameters['img_size']]

        def block_conv(in_chan,out_chan, filter):
            layer = [ nn.Conv2d(in_chan, out_chan, filter),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Dropout2d(0.25)
                      nn.BatchNorm2d(out_chan, 0.8)]
            return layer

        self.model = nn.Sequential(
            *block_conv(parameters['channels'], 16),
            *block_conv(16, 32),
            *block_conv(32, 64),
            *block_conv(64, 128),

            nn.Linear(parameters['n_classes'] + int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_emb_dis(labels)), -1)
        validity = self.model(d_in)
        return validity
