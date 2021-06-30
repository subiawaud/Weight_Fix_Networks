import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import torch
from torchvision.datasets import CocoDetection

class CocoDataModule(pl.LightningDataModule):
        def __init__(self, data_dir: str = 'Datasets/', shuffle_pixels=False, shuffle_labels=False, random_pixels=False):
                super().__init__()
                self.data_dir = data_dir
                self.transform = self.transform_select(shuffle_pixels, random_pixels)
                self.normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                self.target_transform = self.target_transform_select(shuffle_labels)
                self.targets = 10
                self.dims = (3,320,320)
                self.bs = 4
                self.name = 'Coco'

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
                        transforms.RandomResizedCrop([320, 320]),
                        transforms.ToTensor(),
                        self.normalise,
                        random_pixel_transform])

             else:
                return transforms.Compose([
                        transforms.RandomResizedCrop([320, 320]),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(), self.normalise])

        def test_transform(self):
                        return transforms.Compose([
                        transforms.RandomResizedCrop([320, 320]),
                        transforms.ToTensor(), self.normalise])


        def permute_pixels(self, x):
             idx = torch.randperm(x.nelement())
             x = x.view(-1)[idx].view(x.size())
             return x

        def prepare_data(self):
                CocoDetection(self.data_dir, train = True, download = True)
                CocoDetection(self.data_dir, train = False, download = True)

        def setup(self, stage=None):
                if stage == 'fit' or stage is None:
                        coco_full = CocoDetection(self.data_dir, train = True, transform=self.transform, target_transform=self.target_transform, download=True)
                        self.train, self.val = random_split(coco_full, [int(len(cifar_full)*0.9),int(len(cifar_full)*0.1)])
                     #   self.train, self.val = coco_full, cifar_full

                if stage == 'test' or stage is None:
                        self.test = coco(self.data_dir, train = False, transform=self.test_transform(), target_transform=self.target_transform)

        def train_dataloader(self):
                return DataLoader(self.train, batch_size=self.bs, num_workers = 6)

        def val_dataloader(self):
                return DataLoader(self.val, batch_size=self.bs, num_workers = 6)

        def test_dataloader(self):
                return DataLoader(self.test, batch_size=self.bs, num_workers = 6)
