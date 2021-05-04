import torch
from Models.Pretrained_Model_Template import Pretrained_Model_Template
from Datasets import cifar10, mnist, imagenet
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import torchvision.models as models
from PyTorch_CIFAR10.cifar10_models import *
from Models.All_Conv_4 import All_Conv_4
import re
import numpy as np


def get_model(model, data, file_name):
    print(model, data)
    use_sched = True
    if model == 'resnet' and data == 'imnet':
        model = models.resnet18(pretrained=False)
        model.name = 'resnet18'
        model.load_state_dict(torch.load("Models/"+file_name))
        return model

def determine_dataset(data_arg):
    if data_arg == 'cifar10':
        return cifar10.CIFAR10DataModule()
    elif data_arg == 'imnet':
        return imagenet.ImageNet_Module()

def main(args):
    data = determine_dataset(args.dataset)
    data.setup()
    model = get_model(args.model, args.dataset)
    trainer = pl.Trainer(gpus=1, max_epochs =0, num_sanity_val_steps = 0, checkpoint_callback=False)
    trainer.fit(model, data)
    acc = trainer.test(model, ckpt_path=None)[0]['test_acc']
    print(acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name') #0.1, 0.15, 0.2, 0.25, 0.3
    parser.add_argument('--model', default = 'resnet')
    parser.add_argument('--dataset', default='imnet')
    args = parser.parse_args()
    print('This is the args ', args)
    main(args)
