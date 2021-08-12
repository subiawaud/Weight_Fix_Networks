import math
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
    def __init__(self, original_model, max_epochs, data_module, lr, scheduler, opt, hparams=None):
        super(Pretrained_Model_Template, self).__init__()
        dim_in = [3,32,32]
        #self.save_hyperparameters(hparams)
        self.name = original_model.name
        self.pretrained = original_model
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = 0.0005
        self.data_module = data_module
        self.batch_size = data_module.bs
        self.train_size = len(data_module.train_dataloader().dataset)
        self.scheduler = scheduler
        self.opt = opt

    def set_optim(self, max_epochs, lr):
           if self.opt == 'ADAM':
               self.optim = torch.optim.Adam(self.parameters(), lr = self.lr)
           elif self.opt == 'SGD':
               self.optim = torch.optim.SGD(self.parameters(), lr=self.lr,
                      momentum=0.9, weight_decay=self.weight_decay)
           self.scheduler = None # not using one at the moment

    def forward(self, x):
        return self.pretrained(x)
