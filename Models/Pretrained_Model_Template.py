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
    def __init__(self, original_model, max_epochs, data_module, lr, use_sched, opt):
        super(Pretrained_Model_Template, self).__init__()
        dim_in = [3,32,32]
        self.name = original_model.name
        self.pretrained = original_model
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = 1e-4
        self.batch_size = data_module.bs
        self.batch_size = 256
        self.train_size = len(data_module.train_dataloader().dataset)
        self.use_sched = use_sched
        self.opt = opt

    def set_optim(self, max_epochs):
           print('in here ', self.train_size)
           if self.opt == 'ADAM':
               self.optim = torch.optim.Adam(self.parameters(), lr = self.lr)
           elif self.opt == 'SGD':
               self.optim = torch.optim.SGD(self.parameters(), lr=self.lr,
                      momentum=0.9, weight_decay=5e-4)
           else:
               print('NO OPTIMIZER ')

           self.scheduler = None
           if self.use_sched:
              # self.scheduler =   torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max = max_epochs+1)
               self.scheduler =   torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.lr,
                                                                        steps_per_epoch=int(1*self.train_size)//self.batch_size,
                                                                         epochs=max_epochs+1)

    def forward(self, x):
        return self.pretrained(x)
