import os
import numpy as np
import cv2
import torch
import time
import albumentations as albu

import torchvision.transforms as transforms
from PIL import Image
import copy
import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler

from .dataset_tct import InputDataset

from .randaug import RandAugment


def build_loader(args):
#     train_set, train_loader = None, None
#     if args.train_root is not None:
#         train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
#         train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)

#     val_set, val_loader = None, None
#     if args.val_root is not None:
#         val_set = ImageDataset(istrain=False, root=args.val_root, data_size=args.data_size, return_index=True)
#         val_loader = torch.utils.data.DataLoader(val_set, num_workers=1, shuffle=True, batch_size=args.batch_size)
        
#         elif args.dataset == 'TCT':
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])

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

#     print("Loading training data")
#     st = time.time()
#     trainset = InputDataset(args.train_csv, True, train_transform,
#             albu_transform=train_albu_transform)
#     print("Took", time.time() - st)

#     print("Loading validation data")
#     testset = InputDataset(args.val_csv, False, test_transform,
#             albu_transform=test_albu_transform)
    if args.is_test:
        print("Loading test data")
        testset = InputDataset(args.test_csv, False, test_transform,
            albu_transform=test_albu_transform)
        test_loader = DataLoader(testset,
                                 batch_size=args.eval_batch_size,
                                 num_workers=8,
                                 pin_memory=True) if testset is not None else None

        return None, test_loader
    else:
        print("Loading training data")
        trainset = InputDataset(args.train_csv, True, train_transform,
                albu_transform=train_albu_transform)

        print("Loading validation data")
        testset = InputDataset(args.val_csv, False, test_transform,
                albu_transform=test_albu_transform)
    
        train_loader = DataLoader(trainset,
                                  batch_size=args.train_batch_size,
                                  num_workers=24,
                                  drop_last=True,
                                  pin_memory=True)
        test_loader = DataLoader(testset,
                                 batch_size=args.eval_batch_size,
                                 num_workers=8,
                                 pin_memory=True) if testset is not None else None

        return train_loader, test_loader


def get_dataset(args):
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        return train_set
    return None


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 istrain: bool,
                 root: str,
                 data_size: int,
                 return_index: bool = False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        self.data_size = data_size
        self.return_index = return_index

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

        # 448:600
        # 384:510
        # 768:
        if istrain:
            # transforms.RandomApply([RandAugment(n=2, m=3, img_size=data_size)], p=0.1)
            # RandAugment(n=2, m=3, img_size=sub_data_size)
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)


    def getDataInfo(self, root):
        data_infos = []
        folders = os.listdir(root)
        folders.sort() # sort by alphabet
        print("[dataset] class number:", len(folders))
        for class_id, folder in enumerate(folders):
            files = os.listdir(root+folder)
            for file in files:
                data_path = root+folder+"/"+file
                data_infos.append({"path":data_path, "label":class_id})
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.
        
        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)
        
        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, img, label
        
        # return img, sub_imgs, label, sub_boundarys
        return img, label
