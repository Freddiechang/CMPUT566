from os import listdir
from os.path import isfile, join, isdir

import numpy as np
import scipy.io as io
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pytorch_lightning.core.datamodule import LightningDataModule
import os

"""
    A total of 630 target images were chosen from the FIGRIM fine-grained image memorability dataset, 
    by sampling 30 images from each of FIGRIM's 21 indoor and outdoor scene categories.
    website: http://figrim.mit.edu/index_eyetracking.html
    
    image: http://figrim.mit.edu/Targets.zip
    annot: http://figrim.mit.edu/FIXATIONMAPS.zip
    fixlc: http://figrim.mit.edu/FIXATIONLOCS.zip
    
    dataset structure:
        dataset/FIGRIM/
                    -FIXATIONLOCS/[airport_terminal, bathroom,castle, ..., tower]
                    -FIXATIONMAPS/[airport_terminal, bathroom,castle, ..., tower]
                    -Targets/[airport_terminal, bathroom,castle, ..., tower]
"""


class FIGRIM(Dataset):

    @staticmethod
    def load_data(data_directory, error_list=[]):
        images = []
        for directory in sorted(os.listdir(data_directory)):
            # ignore any hidden folder like .DS_Store
            if directory[0] == ".":
                continue
            sub_directory = join(data_directory, directory)
            images.extend(
                sorted([join(sub_directory, f) for f in listdir(sub_directory) if
                        (isfile(join(sub_directory, f)) and f not in error_list)]))
        return images

    def __init__(self, args, mode="test"):
        self.resize = args.resize
        self.normalize = args.normalize
        self.norm_mean = [float(i) for i in args.norm_mean.split('+')]
        self.norm_std = [float(i) for i in args.norm_std.split('+')]
        self.SLICING_INDEX = -65

        # FIGRIM dataset contains some image inside Targets folder without any fixation map and loc
        self.FIGRIM_DATASET_ERROR = ['sun_abpqxslcljhrwmck.jpg',
                                     'sun_bsccnfecifucnavf.jpg',
                                     'sun_auwrraazjwdcjcjg.jpg',
                                     'sun_bdwttbytrbnqyqsk.jpg',
                                     'sun_bjitfqyiepkgfkks.jpg',
                                     'sun_bgdykfpjgudqpzlu.jpg',
                                     '.DS_Store']
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
        self.transform = {"image": transforms.Compose(transform_list["image"]),
                          "annotation": transforms.Compose(transform_list["annotation"])}

        if mode == "test":
            self.images = FIGRIM.load_data(join(data_root, "Targets"), self.FIGRIM_DATASET_ERROR)
            self.images = self.images[self.SLICING_INDEX:]

            self.annotations = FIGRIM.load_data(join(data_root, "FIXATIONMAPS"), self.FIGRIM_DATASET_ERROR)
            self.annotations = self.annotations[self.SLICING_INDEX:]

            self.fixations = FIGRIM.load_data(join(data_root, "FIXATIONLOCS"), self.FIGRIM_DATASET_ERROR)
            self.fixations = self.fixations[self.SLICING_INDEX:]
        else:
            self.images = FIGRIM.load_data(join(data_root, "Targets"), self.FIGRIM_DATASET_ERROR)
            self.images = self.images[:self.SLICING_INDEX]

            self.annotations = FIGRIM.load_data(join(data_root, "FIXATIONMAPS"), self.FIGRIM_DATASET_ERROR)
            self.annotations = self.annotations[:self.SLICING_INDEX]

            self.fixations = FIGRIM.load_data(join(data_root, "FIXATIONLOCS"), self.FIGRIM_DATASET_ERROR)
            self.fixations = self.fixations[:self.SLICING_INDEX]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        annotation_path = self.annotations[idx]
        fixation_path = self.fixations[idx]

        image = Image.open(img_path)
        annotation = Image.open(annotation_path)
        fixation = io.loadmat(fixation_path)["fixLocs"]
        fixation = Image.fromarray(fixation)

        image = self.transform["image"](image)
        # handling grayscale images
        if image.shape[0] == 1:
            image = torch.cat((image, image, image), dim=0)
        annotation = self.transform["annotation"](annotation)
        fixation = self.transform["annotation"](fixation)

        sample = {'image': image, 'annotation': annotation, 'fixation': fixation}
        return sample


class FIGRIMDataModule(LightningDataModule):

    def prepare_data(self, *args, **kwargs):
        pass

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.SPLIT = 200

    def setup(self, stage):
        # split dataset
        if stage == 'fit':
            data_train = FIGRIM(self.args, 'train')
            self.data_train, self.data_val = random_split(data_train, [len(data_train) - self.SPLIT, self.SPLIT])
        if stage == 'test':
            self.data_test = FIGRIM(self.args, 'test')

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
