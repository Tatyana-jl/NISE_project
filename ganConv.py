import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


class Generator(nn.Module):

    def __init__(self, parameters):
        super(Generator, self).__init__()
        
        self.param = parameters
        
        self.label_emb_gen = nn.Embedding(self.param['n_classes'], self.param['n_classes'])
        self.img_shape = [self.param['img_size'], self.param['img_size']]
        self.G_loss=[1]
        
        self.dense = nn.Sequential(
            nn.Linear(self.param['latent_dim'] + self.param['n_classes'], 128*self.param['img_size']**2),
            nn.BatchNorm1d(128*self.param['img_size']**2, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        def block_conv(in_chan, out_chan, filter_s, pad, stride):
            layers = [nn.Conv2d(in_chan, out_chan, filter_s, stride=stride, padding=pad)]
            layers.append(nn.BatchNorm2d(out_chan, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        def block_deconv(in_chan, out_chan, filter_s, pad, stride):
            layers = [nn.ConvTranspose2d(in_chan, out_chan, filter_s, stride=stride, padding=pad)]
            layers.append(nn.BatchNorm2d(out_chan, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.conv = nn.Sequential(           
            *block_conv(128, 128, 3, 1, 1),
            *block_deconv(128, 128, 2, 1, 1),
            *block_conv(128, 128, 4, 1, 1),
            *block_conv(128, 128, 5, 2, 1),
            *block_conv(128, 128, 5, 2, 1),
            nn.Conv2d(128, self.param['channels'], 5, padding=3, stride=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb_gen(labels), noise), -1)
        to_conv = self.dense(gen_input)
        to_conv = to_conv.view(to_conv.shape[0], 128, *self.img_shape)
        img = self.conv(to_conv)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    
    def __init__(self, parameters):
        super(Discriminator, self).__init__()

        self.param = parameters
        self.label_emb_dis = nn.Embedding(self.param['n_classes'], self.param['n_classes'])
        self.img_shape = [self.param['img_size'], self.param['img_size']]
        self.D_loss =[]
        
        def block_conv(in_chan, out_chan, filter_s, pad, stride):
            layers = [ nn.Conv2d(in_chan, out_chan, filter_s, padding=pad, stride=stride),
                      nn.BatchNorm2d(out_chan, 0.8),
                      nn.LeakyReLU(0.2, inplace=True)]
            return layers

        self.conv = nn.Sequential(
            *block_conv(self.param['channels'], 128, 3, 1, 1),
            *block_conv(128, 128, 4, 1, 1),
            *block_conv(128, 128, 4, 1, 1),
            *block_conv(128, 128, 4, 1, 1),
            nn.Dropout(0.4)
        )
        
        dense_input = 25 #dropout is used 
        self.dense = nn.Sequential(nn.Linear(self.param['n_classes']+128*dense_input**2, 1), nn.Sigmoid())

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        conv = self.conv(img)
        dense = torch.cat((conv.view(conv.size(0), -1), self.label_emb_dis(labels)), -1)
        validity = self.dense(dense)
        return validity

