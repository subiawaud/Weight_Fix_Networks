import pytorch_lightning as pl 
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torch

class MNISTDataModule(pl.LightningDataModule):

        def __init__(self, data_dir: str = 'Datasets/', shuffle_pixels=False, shuffle_labels=False, random_pixels=False):
                super().__init__()
                self.data_dir = data_dir
                self.transform = self.transform_select(shuffle_pixels, random_pixels)
                self.target_transform = self.target_transform_select(shuffle_labels) 
                self.dims = (1,28,28)
                self.name = 'MNIST'

        def target_transform_select(self, shuffle_labels = False):
             if shuffle_labels:
                target_transform = lambda y: torch.randint(0, 10, (1,)).item()
                return target_transform
             else:
                return None 

        def transform_select(self, shuffle_pixels = False, random_pixels =False):
             if shuffle_pixels:
                permute_pixels_transform = lambda x: self.permute_pixels(x)
                return transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        permute_pixels_transform])
             elif random_pixels:
                random_pixel_transform = lambda x: torch.rand(x.size())
                return transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        random_pixel_transform])        
             else:
                return transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))])

 
        def permute_pixels(self, x):
             idx = torch.randperm(x.nelement())
             x = x.view(-1)[idx].view(x.size())
             return x

        def prepare_data(self):
                MNIST(self.data_dir, train = True, download = True)
                MNIST(self.data_dir, train = False, download = True)

        def setup(self, stage=None):
                if stage == 'fit' or stage is None:
                        mnist_full = MNIST(self.data_dir, train = True, transform=self.transform, target_transform=self.target_transform)
                        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

                if stage == 'test' or stage is None:
                        self.mnist_test = MNIST(self.data_dir, train = False, transform=self.transform,target_transform=self.target_transform )

        def train_dataloader(self):
                return DataLoader(self.mnist_train, batch_size=128, num_workers = 6)

        def val_dataloader(self):
                return DataLoader(self.mnist_val, batch_size=128, num_workers = 6)
        
        def test_dataloader(self):
                return DataLoader(self.mnist_test, batch_size=128, num_workers = 6)

                                 
