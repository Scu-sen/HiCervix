# -*- coding: utf-8 -*-
# Copyright (c) 2020, Tencent Inc. All rights reserved.
# Author: huye
# Date: 2020-02-25

import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class InputDataset(Dataset):
    """Input Dataset """

    def __init__(self, data_csv_file, train=True, transform=None,
            target_transform=None, albu_transform=None):
        """
        Args:
            data_csv_file: csv_file, [image_path, class_id]
            train: bool
            transform: image transform
            albu_transform: albumentations lib support
        """
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.albu_transform = albu_transform

        # load data
        df = pd.read_csv(data_csv_file)
        self.data = []
#         level_1_dict = {'negative':0,'ASC':1, 'AGC':2,'microbe':3}
        for n in range(len(df)):
            row = df.iloc[n]
            image_path = row["image_path"]
            class_id = int(row["level_3_id"])
            self.data.append((image_path, class_id,))
        if self.train:
            random.shuffle(self.data)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        # read image
        #print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.albu_transform is not None:
           img = self.albu_transform(image=img)["image"]

        #img = Image.open(img_path) # RGB
        #img = img.resize((299, 299))
        img = Image.fromarray(img)
        # transform
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


