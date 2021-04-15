import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch

class CIFAR10DataModule(pl.LightningDataModule):
        def __init__(self, data_dir: str = 'Datasets/', shuffle_pixels=False, shuffle_labels=False, random_pixels=False):
                super().__init__()
                self.data_dir = data_dir
                self.normalise = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                self.transform = self.transform_select(shuffle_pixels, random_pixels)
                self.target_transform = self.target_transform_select(shuffle_labels)
                self.targets = 10
                self.dims = (3,32,32)
                self.bs = 256
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
                        self.normalise,
                        permute_pixels_transform])

             elif random_pixels:
                random_pixel_transform = lambda x: torch.rand(x.size())
                return transforms.Compose([
                        transforms.ToTensor(),
                        self.normalise,
                        random_pixel_transform])

             else:
                return transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        self.normalise])
        def test_transform(self):
                        return transforms.Compose([
                        transforms.ToTensor(),
                        self.normalise])



        def permute_pixels(self, x):
             idx = torch.randperm(x.nelement())
             x = x.view(-1)[idx].view(x.size())
             return x

        def prepare_data(self):
                CIFAR10(self.data_dir, train = True, download = True)
                CIFAR10(self.data_dir, train = False, download = True)

        def setup(self, stage=None):
                if stage == 'fit' or stage is None:
                        cifar_full = CIFAR10(self.data_dir, train = True, transform=self.transform, target_transform=self.target_transform, download=True)
                        self.train, self.val = random_split(cifar_full, [45000, 5000])

                if stage == 'test' or stage is None:
                        self.test = CIFAR10(self.data_dir, train = False, transform=self.test_transform(), target_transform=self.target_transform)

        def train_dataloader(self):
                return DataLoader(self.train, batch_size=self.bs, num_workers = 6)

        def val_dataloader(self):
                return DataLoader(self.val, batch_size=self.bs, num_workers = 6)

        def test_dataloader(self):
                return DataLoader(self.test, batch_size=self.bs, num_workers = 6)
