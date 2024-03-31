""" CUB-200-2011 (Bird) Dataset"""
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
from utils import get_transform

import numpy as np
import cv2
import torch
import time
import albumentations as albu

import torchvision.transforms as transforms
from PIL import Image
import copy
import torch

from .tct import InputDataset

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler

DATAPATH = './CUB_200_2011'
image_path = {}
image_label = {}

def get_dataset(config):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.ColorJitter(brightness=10, contrast=10, saturation=20, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       ])


    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_albu_transform = albu.Compose([
        albu.PadIfNeeded(min_height=1000, min_width=1000,
            border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), always_apply=True),
        #albu.Rotate(limit=180, interpolation=cv2.INTER_LINEAR,
         #   border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.8),
        #albu.CenterCrop(224, 224, always_apply=True),
        albu.RandomCrop(700, 700, always_apply=True),
        albu.RandomBrightnessContrast(),
        albu.HueSaturationValue(),
        albu.OneOf([
            albu.IAAAdditiveGaussianNoise(p=0.5),
            albu.GaussNoise(p=0.5),
            ], p=0.5),
        albu.OneOf([
            albu.Blur(blur_limit=7, p=0.5),
            albu.MedianBlur(blur_limit=7, p=0.5),
            #albu.MotionBlur(blur_limit=7, p=0.5),
            ], p=0.5),
        #albu.Resize(448, 448, interpolation=cv2.INTER_LINEAR, always_apply=True),
        albu.Resize(384, 384, interpolation=cv2.INTER_LINEAR, always_apply=True),
        ])

    test_albu_transform = albu.Compose([
        albu.PadIfNeeded(min_height=1000, min_width=1000,
            border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), always_apply=True),
        albu.CenterCrop(700, 700, always_apply=True),
        #albu.Resize(448, 448, interpolation=cv2.INTER_LINEAR, always_apply=True),
        albu.Resize(384, 384, interpolation=cv2.INTER_LINEAR, always_apply=True),
        ])
    if config.is_test:
        print("Loading test data")
        testset = InputDataset(config.test_csv, False, test_transform,
            albu_transform=test_albu_transform)
        return testset
    else:
        print("Loading training data")
        trainset = InputDataset(config.train_csv, True, train_transform,
                albu_transform=train_albu_transform)

        print("Loading validation data")
        testset = InputDataset(config.val_csv, False, test_transform,
                albu_transform=test_albu_transform)
        return trainset, testset


class TCTDataset(Dataset):
    """
    # Description:
        Dataset for retrieving CUB-200-2011 images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', resize=500):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.image_id = []
        self.num_classes = 200

        # get image path from images.txt
        with open(os.path.join(DATAPATH, 'images.txt')) as f:
            for line in f.readlines():
                id, path = line.strip().split(' ')
                image_path[id] = path

        # get image label from image_class_labels.txt
        with open(os.path.join(DATAPATH, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                id, label = line.strip().split(' ')
                image_label[id] = int(label)

        # get train/test image id from train_test_split.txt
        with open(os.path.join(DATAPATH, 'train_test_split.txt')) as f:
            for line in f.readlines():
                image_id, is_training_image = line.strip().split(' ')
                is_training_image = int(is_training_image)

                if self.phase == 'train' and is_training_image:
                    self.image_id.append(image_id)
                if self.phase in ('val', 'test') and not is_training_image:
                    self.image_id.append(image_id)

        # transform
        self.transform = get_transform(self.resize, self.phase)

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]

        # image
        image = Image.open(os.path.join(DATAPATH, 'images', image_path[image_id])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, image_label[image_id] - 1  # count begin from zero

    def __len__(self):
        return len(self.image_id)


if __name__ == '__main__':
    ds = BirdDataset('train')
    print(len(ds))
    for i in range(0, 10):
        image, label = ds[i]
        print(image.shape, label)
