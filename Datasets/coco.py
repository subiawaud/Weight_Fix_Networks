import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import torch
from torchvision.datasets import CocoDetection

class CocoDataModule(pl.LightningDataModule):
        def __init__(self, data_dir: str = 'Datasets/', shuffle_pixels=False, shuffle_labels=False, random_pixels=False):
                super().__init__()
                self.data_year = 2017
                self.data_dir = '/scratch/cc2u18/data/coco/images/'
                self.data_json = '/scratch/cc2u18/data/coco/annotations/'
                self.normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                self.transform = self.transform_select(shuffle_pixels, random_pixels)
                self.target_transform = self.target_transform_select(shuffle_labels)
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

        def setup(self, stage=None):
                if stage == 'fit' or stage is None:
                        coco = CocoDetection(root=self.data_dir+f'/train{self.data_year}', annFile=self.data_json+f'/instances_train{self.data_year}.json', transform=self.transform, target_transform=self.target_transform)
                        s = len(coco)
                        self.train, self.val = random_split(coco, [int(0.9*s)+1, int(0.1*s)])

                if stage == 'test' or stage is None:
                        self.test  = CocoDetection(root=self.data_dir+f'/val{self.data_year}', annFile=self.data_json+f'/instances_val{self.data_year}.json', transform=self.transform, target_transform=self.target_transform)

        def train_dataloader(self):
                return DataLoader(self.train, batch_size=self.bs, num_workers = 6)

        def val_dataloader(self):
                return DataLoader(self.val, batch_size=self.bs, num_workers = 6)

        def test_dataloader(self):
                return DataLoader(self.test, batch_size=self.bs, num_workers = 6)
