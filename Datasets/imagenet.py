import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet, ImageFolder
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
import io
import os
import pickle
import h5py
from PIL import Image
from torchvision.datasets import VisionDataset


class ImageNetHDF5(VisionDataset):
    def __init__(self, root, cache_size=500, transform=None):
        super(ImageNetHDF5, self).__init__(root, transform=transform, target_transform=None)

        #self.dest = pickle.load(open(os.path.join(root, 'dest.p'), 'rb'))
        self.cache = {}
        self.cache_size = cache_size

        targets = sorted(list(filter(lambda f: '.hdf5' in f, os.listdir(root))))
        print(targets)
        self.targets = {f[:-5]: i for i, f in enumerate(targets)}
        self.fill_cache()
        self.dest = self.cache.keys()

    def load(self, file, i):
        with h5py.File(os.path.join(self.root, file + '.hdf5'), 'r') as f:
            return f['data'][i]

    def fill_cache(self):
        print('Filling cache')
        files = (f[:-5] for f in list(filter(lambda f: '.hdf5' in f, os.listdir(self.root)))[:self.cache_size])
        for file in files:
            with h5py.File(os.path.join(self.root, file + '.hdf5'), 'r') as f:
                self.cache[file] = list(f['data'])
        print('Done')

    def load_from_cache(self, file, i):
        if file in self.cache:
            return self.cache[file][i]
        return self.load(file, i)

    def __getitem__(self, index):
        dest, i = self.dest[index]

        sample = self.load_from_cache(dest, i)

        sample = Image.open(io.BytesIO(sample))
        sample = sample.convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.targets[dest]

    def __len__(self):
        return len(self.dest)


class ImageNet_Module(pl.LightningDataModule):
        def __init__(self, data_dir = 'Datasets/', shuffle_pixels=False, shuffle_labels=False, random_pixels=False):
                super().__init__()
                self.data_dir = data_dir
                self.mean = [0.485, 0.456, 0.406]
                self.std = [0.229, 0.224, 0.225]
                self.normalise = transforms.Normalize(mean=self.mean, std=self.std)
                self.transform = self.transform_select(shuffle_pixels, random_pixels)
                self.target_transform = self.target_transform_select(shuffle_labels)
                self.targets = 1000
                self.dims = (3,224,224)
                self.bs = 256
                self.name = 'ImageNet'

        def target_transform_select(self, shuffle_labels = False):
             if shuffle_labels:
                target_transform = lambda y: torch.randint(0, self.targets, (1,)).item()
                return target_transform
             else:
                return None

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
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        self.normalise])


        def permute_pixels(self, x):
             idx = torch.randperm(x.nelement())
             x = x.view(-1)[idx].view(x.size())
             return x

        def setup(self, stage=None):
                if stage == 'fit' or stage is None:
                        im_full = ImageNetHDF5(self.data_dir + 'train',  transform=self.transform)
                        train_s = int(len(im_full)*0.9)
                        val_s = len(im_full) - train_s
                        self.train, self.val = random_split(im_full, [train_s, val_s])
                        print(self.train)
                        print(self.val)

                if stage == 'test' or stage is None:
                        self.test = ImageNetHDF5(self.data_dir + 'train', transform=self.transform)
#                        self.test = ImageNetHDF5(self.data_dir + 'val', transform=self.transform)

        def train_dataloader(self):
                return DataLoader(self.train, batch_size=self.bs, num_workers = 6)

        def val_dataloader(self):
                return DataLoader(self.val, batch_size=self.bs, num_workers = 6)

        def test_dataloader(self):
                return DataLoader(self.test, batch_size=self.bs, num_workers = 6)
