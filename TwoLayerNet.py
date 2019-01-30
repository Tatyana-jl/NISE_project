import numpy as np
import torch
import torch.nn as nn

class twoLayerNet(nn.Module):
    
    def __init__(self, D_in, H, D_out):
        super(twoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0) # Linear layer + ReLu
        y_pred = self.linear2(h_relu)
        return y_pred