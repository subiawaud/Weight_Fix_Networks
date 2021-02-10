import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import torchvision.models as models


class ImageNet_Module(pl.LightningDataModule):
        def __init__(self, data_dir: str = 'Datasets/', shuffle_pixels=False, shuffle_labels=False, random_pixels=False):
                super().__init__()
                self.data_dir = data_dir
                self.mean = [0.485, 0.456, 0.406]
                self.std = [0.229, 0.224, 0.225]
                self.normalise = transforms.Normalize(mean=self.mean, std=self.std)
                self.transform = self.transform_select(shuffle_pixels, random_pixels)
                self.target_transform = self.target_transform_select(shuffle_labels)
                self.targets = 1000
                self.dims = (3,224,224)
                self.bs = 128
                self.name = 'ImageNet'

        def target_transform_select(self, shuffle_labels = False):
             if shuffle_labels:
                target_transform = lambda y: torch.randint(0, self.targets, (1,)).item()
                return target_transform
             else:
                return None
                                                    transforms.Resize((256, 256)),
                                                    transforms.CentreCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(self.mean, self.std)])

        def transform_select(self, shuffle_pixels = False, random_pixels=False):
             if shuffle_pixels:
                permute_pixels_transform = lambda x: self.permute_pixels(x)
                return transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.CentreCrop(224),
                        transforms.ToTensor(),
                        self.normalise,
                        permute_pixels_transform])

             elif random_pixels:
                random_pixel_transform = lambda x: torch.rand(x.size())
                return transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.CentreCrop(224),
                        transforms.ToTensor(),
                        self.normalise,
                        random_pixel_transform])

             else:
                return transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.CentreCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        self.normalise])


        def permute_pixels(self, x):
             idx = torch.randperm(x.nelement())
             x = x.view(-1)[idx].view(x.size())
             return x

        def prepare_data(self):
                ImageNet(self.data_dir, train = True, download = True)
                ImageNet(self.data_dir, train = False, download = True)

        def setup(self, stage=None):
                if stage == 'fit' or stage is None:
                        im_full = ImageNet(self.data_dir, train = True, transform=self.transform, target_transform=self.target_transform)
                        self.train, self.val = random_split(im_full, [45000, 5000])

                if stage == 'test' or stage is None:
                        self.test = ImageNet(self.data_dir, train = False, transform=self.transform, target_transform=self.target_transform)

        def train_dataloader(self):
                return DataLoader(self.train, batch_size=self.bs, num_workers = 6)

        def val_dataloader(self):
                return DataLoader(self.val, batch_size=self.bs, num_workers = 6)

        def test_dataloader(self):
                return DataLoader(self.test, batch_size=self.bs, num_workers = 6)
