import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet, ImageFolder
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
import io
import os
import pickle
from PIL import Image
from torchvision.datasets import VisionDataset



class ImageNet_Module(pl.LightningDataModule):
        def __init__(self, data_dir = 'Datasets/', shuffle_pixels=False, shuffle_labels=False, random_pixels=False):
                super().__init__()
                self.data_dir = '/ECSssd/data_sets/imagenet_2012'
                self.mean = [0.485, 0.456, 0.406]
                self.std = [0.229, 0.224, 0.225]
                self.normalise = transforms.Normalize(mean=self.mean, std=self.std)
                self.transform = self.transform_select(shuffle_pixels, random_pixels)
                self.test_trans = self.test_transform()
                self.target_transform = self.target_transform_select(shuffle_labels)
                self.targets = 1000
                self.dims = (3,224,224)
                self.bs = 256
#                torch.distributed.init_process_group('nccl')
                self.name = 'ImageNet'

        def target_transform_select(self, shuffle_labels = False):
             if shuffle_labels:
                target_transform = lambda y: torch.randint(0, self.targets, (1,)).item()
                return target_transform
             else:
                return None

        def test_transform(self):
                return transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        self.normalise])


        def transform_select(self, shuffle_pixels = False, random_pixels=False):
             if shuffle_pixels:
                permute_pixels_transform = lambda x: self.permute_pixels(x)
                return transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        self.normalise,
                        permute_pixels_transform])

             elif random_pixels:
                random_pixel_transform = lambda x: torch.rand(x.size())
                return transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        self.normalise,
                        random_pixel_transform])

             else:
                return transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.RandomCrop((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        self.normalise])


        def permute_pixels(self, x):
             idx = torch.randperm(x.nelement())
             x = x.view(-1)[idx].view(x.size())
             return x

        def setup(self, stage=None):
                if stage == 'fit' or stage is None:
                        im_full = ImageFolder(self.data_dir + '/train',  transform=self.transform)
                        train_s = int(len(im_full)*0.9)
                        val_s = len(im_full) - train_s
                        self.train, self.val = random_split(im_full, [train_s, val_s])

                if stage == 'test' or stage is None:
                        self.test = ImageFolder(self.data_dir + '/val', transform=self.test_trans)

        def train_dataloader(self):
#'                sampler = torch.utils.data.distributed.DistributedSampler(self.train)
                return DataLoader(self.train, batch_size=self.bs, num_workers = 8, pin_memory=True)#, sampler=sampler)

        def val_dataloader(self):
#                sampler = torch.utils.data.distributed.DistributedSampler(self.val)
                return DataLoader(self.val, batch_size=self.bs, num_workers = 8, pin_memory=True)#, sampler=sampler)

        def test_dataloader(self):
#                sampler = torch.utils.data.distributed.DistributedSampler(self.test)
                return DataLoader(self.test, batch_size=self.bs, num_workers = 8, pin_memory=True)#, sampler=sampler)
