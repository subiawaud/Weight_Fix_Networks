import math
import seaborn as sns
import copy
import torch
import numpy as np
from Models.Weight_Fix_Base import Weight_Fix_Base
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn.utils.prune as prune
from pytorch_lightning.metrics import functional as FM
import matplotlib.pyplot as plt


class Pretrained_Model_Template(Weight_Fix_Base):
    def __init__(self, original_model, max_epochs, data_module ):
        super(Pretrained_Model_Template, self).__init__()
        dim_in = [3,32,32]
        self.pretrained = original_model
        self.max_epochs = max_epochs
        self.lr = 1e-5
        self.weight_decay = 1e-4
        self.batch_size = data_module.bs
        self.train_size = len(data_module.train_dataloader().dataset)
        print(self.train_size, ' THE TRAIN SIZE')

    def set_optim(self, max_epochs):
           self.optim = torch.optim.Adam(self.parameters(), lr = self.lr)
           self.scheduler = None
           self.scheduler =   torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.lr,
                                                                        steps_per_epoch=int(1*self.train_size)//self.batch_size,
                                                                         epochs=max_epochs+1)

    def forward(self, x):
        return self.pretrained(x)
