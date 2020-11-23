import math
import seaborn as sns
import copy
import torch 
import numpy as np
from Models.Weight_Fix_Base import Weight_Fix_Base
import torch.nn as nn
from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
import torch.nn.functional as F
import pytorch_lightning as pl 
import torch.nn.utils.prune as prune
from pytorch_lightning.metrics import functional as FM
import matplotlib.pyplot as plt


class All_Conv_4(Weight_Fix_Base):
    def __init__(self):
        super(All_Conv_4, self).__init__()
        dim_in = [3,32,32]
        self.conv1 = nn.Conv2d(dim_in[0], 64, kernel_size=3, stride = 1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride = 1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride = 1)
        self.dim_convert = {28:4, 32:5}
        self.fc1 = nn.Linear(128 * self.dim_convert[dim_in[1]]**2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.final = nn.Linear(256, 10)
        self.group_version = False
        self.name = 'All_Conv_4'
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.forward_order = [self.conv1, F.relu, self.conv2, F.relu, self.pool, self.conv3, F.relu, self.conv4 ,F.relu, self.pool]
        self.lr = 3e-4
        
    def set_optim(self):
           self.optim = torch.optim.Adam(self.parameters(), lr = self.lr)
           self.scheduler = torch.optim.lr_scheduler.StepLR(
                 self.optim, 
                 step_size=50, 
                 gamma=0.1) 
                 
    def forward(self, x):
        for f in self.forward_order:
            x = f(x) 
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.final(x)
        return F.log_softmax(x, dim = 1)

