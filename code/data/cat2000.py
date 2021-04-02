from os import listdir, walk
from os.path import isfile, join, isdir

import numpy as np
import scipy.io as io
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pytorch_lightning.core.datamodule import LightningDataModule


class CAT2000(Dataset):
    def __init__(self, args, mode="test"):
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
        self.transform = {"image": transforms.Compose(transform_list["image"]),
                          "annotation": transforms.Compose(transform_list["annotation"])}

        tmp_path = join(data_root, 'Stimuli')
        category_folder_name = listdir(tmp_path)
        self.images = []
        for folder in category_folder_name:
            tmp_path_next = join(tmp_path, folder)
            file_list = [join(tmp_path_next, f) for f in listdir(tmp_path_next) if isfile(join(tmp_path_next, f))]
            self.images.extend(file_list)
        self.images = sorted(self.images)

        tmp_path = join(data_root, 'FIXATIONMAPS')
        category_folder_name = listdir(tmp_path)
        self.annotations = []
        for folder in category_folder_name:
            tmp_path_next = join(tmp_path, folder)
            file_list = [join(tmp_path_next, f) for f in listdir(tmp_path_next) if isfile(join(tmp_path_next, f))]
            self.annotations.extend(file_list)
        self.annotations = sorted(self.annotations)

        tmp_path = join(data_root, 'FIXATIONLOCS')
        category_folder_name = listdir(tmp_path)
        self.fixations = []
        for folder in category_folder_name:
            tmp_path_next = join(tmp_path, folder)
            file_list = [join(tmp_path_next, f) for f in listdir(tmp_path_next) if isfile(join(tmp_path_next, f))]
            self.fixations.extend(file_list)
        self.fixations = sorted(self.fixations)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        annotation_path = self.annotations[idx]
        fixation_path = self.fixations[idx]

        image = Image.open(img_path)
        annotation = Image.open(annotation_path)
        # Matlab file for fixation
        # fixation = Image.open(fixation_path)
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


class CAT2000DataModule(LightningDataModule):

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
        self.data_test = CAT2000(self.args)

    # return the test dataloader for each split
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


# test code
'''
tmp_path = "trainSet\\FIXATIONLOCS"
category_folder_name = listdir(tmp_path)
fixations = []
for folder in category_folder_name:
    tmp_path_next = join(tmp_path, folder)
    file_list = [join(tmp_path_next, f) for f in listdir(tmp_path_next) if isfile(join(tmp_path_next, f))]
    fixations.extend(file_list)
fixations = sorted(fixations)


class Args:
    def __init__(self):
        self.data_root = "trainSet"
        self.normalize = "False"
        self.norm_mean = "0.485 + 0.456 + 0.406"
        self.norm_std = "0.229 + 0.224 + 0.225"
        self.resize = True
        self.height = 256
        self.width = 256
        self.batch_size = 16
        self.dataloader_workers = 6
        self.shuffle = False
        self.prefetch = 2


args = Args()
aa = CAT2000(args)

fixation_path = aa.fixations[0]
fixation = io.loadmat(fixation_path)["fixLocs"]
fixation = Image.fromarray(fixation)
fixation = aa.transform["annotation"](fixation)

'''