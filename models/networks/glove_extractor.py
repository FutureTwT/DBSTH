import torch
import torch.nn as nn

import os
import pickle as pkl

Glove_ROOT = os.path.dirname(os.path.dirname(__file__)) + '/weights/'

# Check in 2023-03-29
class Glove(nn.Module):
    def __init__(self, name):
        super(Glove, self).__init__()
        glove_weights = pkl.load(open(Glove_ROOT + name + '_bow_weights.pkl', 'rb'))['weights']  # [W, D]
        self.glove_weights = torch.Tensor(glove_weights).unsqueeze(dim=0).cuda() # [1, W, D]

    def forward(self, x):
        x = torch.unsqueeze(x, dim=2).expand(-1, -1, self.glove_weights.shape[2]) # [N, W, D]
        x = torch.mul(x, self.glove_weights) # element-wise
        return x

