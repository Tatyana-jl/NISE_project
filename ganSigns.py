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

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(parameters['latent_dim']+parameters['n_classes'], 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb_gen(labels), noise), -1)
        # print(self.label_emb_gen(labels).shape)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, parameters):
        super(Discriminator, self).__init__()

        self.label_emb_dis = nn.Embedding(parameters['n_classes'], parameters['n_classes'])
        self.img_shape = [parameters['img_size'], parameters['img_size']]
        self.model = nn.Sequential(
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


