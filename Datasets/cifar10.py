import pytorch_lightning as pl 
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch

class CIFAR10DataModule(pl.LightningDataModule):
        def __init__(self, data_dir: str = 'Datasets/', shuffle_pixels=False, shuffle_labels=False, random_pixels=False):
                super().__init__()
                self.data_dir = data_dir
                self.old_normalise = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                self.noise_normalise = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
                self.transform = self.transform_select(shuffle_pixels, random_pixels)
                self.target_transform = self.target_transform_select(shuffle_labels)
                self.dims = (3,32,32)
                self.bs = 64
                self.name = 'CIFAR10'

        def target_transform_select(self, shuffle_labels = False):
             if shuffle_labels:
                target_transform = lambda y: torch.randint(0, 10, (1,)).item()
                return target_transform
             else:
                return None 

        def transform_select(self, shuffle_pixels = False, random_pixels=False):
             if shuffle_pixels:
                permute_pixels_transform = lambda x: self.permute_pixels(x)
                return transforms.Compose([
                        transforms.ToTensor(),
                        self.noise_normalise,
                        permute_pixels_transform])

             elif random_pixels:
                random_pixel_transform = lambda x: torch.rand(x.size())
                return transforms.Compose([
                        transforms.ToTensor(),
                        self.noise_normalise,
                        random_pixel_transform])  

             else:
                return transforms.Compose([
                        transforms.RandomVerticalFlip(0.1), 
                        transforms.RandomHorizontalFlip(0.3),
                        transforms.RandomAffine(degrees = 15, translate=[0.1, 0.1], scale=(0.95, 1.05)),
                        transforms.ToTensor(),
                        self.noise_normalise])

 
        def permute_pixels(self, x):
             idx = torch.randperm(x.nelement())
             x = x.view(-1)[idx].view(x.size())
             return x

        def prepare_data(self):
                CIFAR10(self.data_dir, train = True, download = True)
                CIFAR10(self.data_dir, train = False, download = True)

        def setup(self, stage=None):
                if stage == 'fit' or stage is None:
                        cifar_full = CIFAR10(self.data_dir, train = True, transform=self.transform, target_transform=self.target_transform)
                        self.train, self.val = random_split(cifar_full, [45000, 5000])

                if stage == 'test' or stage is None:
                        self.test = CIFAR10(self.data_dir, train = False, transform=self.transform, target_transform=self.target_transform)

        def train_dataloader(self):
                return DataLoader(self.train, batch_size=self.bs, num_workers = 6)

        def val_dataloader(self):
                return DataLoader(self.val, batch_size=self.bs, num_workers = 6)
        
        def test_dataloader(self):
                return DataLoader(self.test, batch_size=self.bs, num_workers = 6)

                                 
