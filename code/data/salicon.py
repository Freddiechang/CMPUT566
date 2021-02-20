from os import listdir
from os.path import isfile, join, isdir

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image



class SALICON(Dataset):
    def __init__(self, args, mode="train"):
        self.resize = args.resize
        self.normalize = args.normalize
        data_root = args.data_root

        # transform for image and annotation/fixation map, respectively
        transform_list = {"image": [], "annotation": []}
        if self.resize:
            transform_list["image"].append(transforms.Resize((args.height, args.width)))
            transform_list["annotation"].append(transforms.Resize((args.height, args.width), Image.NEAREST))
        if self.normalize:
            transform_list["image"].append(transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1]))
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
