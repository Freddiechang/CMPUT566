from os import listdir
from os.path import isfile, join, isdir

import numpy as np
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
from pytorch_lightning.core.datamodule import LightningDataModule



class SALICON(Dataset):
    def __init__(self, args, mode="train"):
        self.resize = args.resize
        self.normalize = args.normalize
        self.norm_mean = [float(i) for i in args.norm_mean.split('+')]
        self.norm_std = [float(i) for i in args.norm_std.split('+')]
        data_root = args.data_root

        # transform for image and annotation/fixation map, respectively
        transform_list = {"image": [], "annotation": []}
        if self.resize:
            transform_list["image"].append(transforms.Resize((args.height, args.width)))
            transform_list["annotation"].append(transforms.Resize((args.height, args.width), Image.NEAREST))
        if self.normalize:
            transform_list["image"].append(transforms.Normalize(self.norm_mean, self.norm_std))
        transform_list["image"].append(transforms.ToTensor())
        transform_list["annotation"].append(transforms.ToTensor())
        self.transform = {"image": transforms.Compose(transform_list["image"]), "annotation": transforms.Compose(transform_list["annotation"])}

        if mode == "test":
            tmp_path = join(data_root, 'images', 'val')
            self.images = sorted([join(tmp_path, f) for f in listdir(tmp_path) if isfile(join(tmp_path, f))])
            self.images = self.images[-1500:]

            tmp_path = join(data_root, 'annotations', 'val')
            self.annotations = sorted([join(tmp_path, f) for f in listdir(tmp_path) if isfile(join(tmp_path, f))])
            self.annotations = self.annotations[-1500:]

            tmp_path = join(data_root, 'fixations', 'val')
            self.fixations = sorted([join(tmp_path, f) for f in listdir(tmp_path) if isfile(join(tmp_path, f))])
            self.fixations = self.fixations[-1500:]
        else:
            tmp_path = join(data_root, 'images', 'train')
            self.images = sorted([join(tmp_path, f) for f in listdir(tmp_path) if isfile(join(tmp_path, f))])
            tmp_path = join(data_root, 'images', 'val')
            self.images += sorted([join(tmp_path, f) for f in listdir(tmp_path) if isfile(join(tmp_path, f))])[:-1500]

            tmp_path = join(data_root, 'annotations', 'train')
            self.annotations = sorted([join(tmp_path, f) for f in listdir(tmp_path) if isfile(join(tmp_path, f))])
            tmp_path = join(data_root, 'annotations', 'val')
            self.annotations += sorted([join(tmp_path, f) for f in listdir(tmp_path) if isfile(join(tmp_path, f))])[:-1500]

            tmp_path = join(data_root, 'fixations', 'train')
            self.fixations = sorted([join(tmp_path, f) for f in listdir(tmp_path) if isfile(join(tmp_path, f))])
            tmp_path = join(data_root, 'fixations', 'val')
            self.fixations += sorted([join(tmp_path, f) for f in listdir(tmp_path) if isfile(join(tmp_path, f))])[:-1500]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        annotation_path = self.annotations[idx]
        fixation_path = self.fixations[idx]



        image = Image.open(img_path)
        annotation = Image.open(annotation_path)
        fixation = Image.open(fixation_path)

        image = self.transform["image"](image)
        annotation = self.transform["annotation"](annotation)
        fixation = self.transform["annotation"](fixation)

        sample = {'image': image, 'annotation': annotation, 'fixation': fixation}
        return sample

class SALICONDataModule(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        if stage == 'fit':
            data_train = SALICON(self.args, 'train')
            self.data_train, self.data_val = random_split(data_train, [10500, 3000])
        if stage == 'test':
            self.data_test = SALICON(self.args, 'test')

    # return the dataloader for each split
    def train_dataloader(self):
        data_train = DataLoader(self.data_train, 
                                batch_size=self.args.batch_size,
                                shuffle=self.args.shuffle,
                                num_workers=self.args.dataloader_workers,
                                pin_memory=True,
                                drop_last=False,
                                prefetch_factor=self.args.prefetch
                                )
        return data_train

    def val_dataloader(self):
        data_val = DataLoader(self.data_val, 
                                batch_size=self.args.batch_size,
                                shuffle=self.args.shuffle,
                                num_workers=self.args.dataloader_workers,
                                pin_memory=True,
                                drop_last=False,
                                prefetch_factor=self.args.prefetch
                                )
        return data_val

    def test_dataloader(self):
        data_test = DataLoader(self.data_test, 
                                batch_size=self.args.batch_size,
                                shuffle=self.args.shuffle,
                                num_workers=self.args.dataloader_workers,
                                pin_memory=True,
                                drop_last=False,
                                prefetch_factor=self.args.prefetch
                                )
        return data_test
