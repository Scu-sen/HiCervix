# -*- coding: utf-8 -*-
# Copyright (c) 2020, Tencent Inc. All rights reserved.
# Author: huye
# Date: 2020-02-26

import os
import os.path as osp
import time
import datetime

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import albumentations as albu
import pickle
import timm
import models
from datasets import InputDataset

# class_orders = ['negative','ASC', 'AGC','microbe']
# class_orders = ['其他正常细胞','宫颈管细胞','修复细胞','化生细胞','糖原溶解细胞','萎缩性改变','子宫内膜细胞','深染细胞团',
#                'ASC-US','LSIL','ASC-H','HSIL', 'SCC',
#                 'AGC-FN', #'AGC','AGC-NOS', 'ADC',
#                 '非典型颈管腺细胞','非典型子宫内膜细胞', '颈管腺癌','子宫内膜腺癌',
#                '念珠菌','放线菌','滴虫','疱疹病毒感染','细菌性阴道病']
# class_orders = ['其他正常细胞','宫颈管细胞','修复细胞','化生细胞','糖原溶解细胞','萎缩性改变','子宫内膜细胞','深染细胞团',
#                'ASC-US','LSIL','ASC-H','HSIL', 'SCC',
#                'AGC','AGC-NOS','AGC-FN','ADC',
#                 '非典型颈管腺细胞','非典型子宫内膜细胞', '颈管腺癌','子宫内膜腺癌',
#                '念珠菌','放线菌','滴虫','疱疹病毒感染','细菌性阴道病']

# id_to_class = dict(zip(range(len(class_orders)), class_orders))
# added_classes = []
# class_to_id = {id_to_class[k]: k for k in id_to_class}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_data(test_csv):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_trandform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=10, contrast=10, saturation=20, hue=0.1),
        transforms.ToTensor(),
        normalize,
        ])

    test_trandform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])

    albu_transform = albu.Compose([
        albu.PadIfNeeded(min_height=1000, min_width=1000,
            border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), always_apply=True),
        albu.CenterCrop(700, 700, always_apply=True),
        albu.Resize(384, 384, interpolation=cv2.INTER_LINEAR, always_apply=True),
        ])

    dataset_test = InputDataset(test_csv, False, test_trandform, albu_transform=albu_transform)

    return dataset_test

def main(args):
    print(args)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    torch.backends.cudnn.benchmark = True

    dataset_test = load_data(args.test_csv)
    data_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    if args.model=='swint':
        model = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True, num_classes=args.num_classes)
    else:
        model = models.__dict__[args.model](pretrained=args.pretrained,
            num_classes=args.num_classes)
    model = model.cuda()
    model.eval()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    start_time = time.time()

    probs = []
    pred_classes = []
    test_df = pd.read_csv(args.test_csv)
    all_probs = []

    correct = 0
    feature_list = []
    target_list = []
    with torch.no_grad():
        for image, target in tqdm(data_loader):
            image = image.cuda()
           # print(target.shape)
            target_list.append(target.numpy())
            target = target.cuda()
#             output, feature = model(image)
            output = model(image)
            output = F.softmax(output, dim=1)
            output_cls = output.argmax(dim=1)
            output_prob = output.max(dim=1).values
#             feature_list.append(feature.cpu().numpy())
            probs.extend(list(output_prob.cpu().numpy()))
            pred_classes.extend(list(output_cls.cpu().numpy()))
            all_probs.append(output.cpu().numpy())

            correct += output_cls.eq(target).sum().item()
    print("Test accuracy:", correct/len(test_df))
#     features = np.vstack(feature_list)
    targets = np.hstack(target_list)
    targets = targets.T
    #print(features.shape)
#     print(targets.shape)
#     feature_targets = np.hstack((targets[:,np.newaxis],features))
    #np.save(os.path.join(args.output_dir,'val_features.npy'),feature_targets)
#     conf_mat = confusion_matrix(test_df["class_id"], pred_classes, labels=list(id_to_class.keys()))
#     p, r, f1score, _ = precision_recall_fscore_support(test_df["class_id"], pred_classes, labels=list(id_to_class.keys()))

#     all_data = np.concatenate([conf_mat.astype(np.float32),
#         p.reshape(-1, 1), r.reshape(-1, 1), f1score.reshape(-1, 1)], 1)

#     eval_df = pd.DataFrame(all_data, index=list(id_to_class.values())+added_classes,
#             columns=list(id_to_class.values()) + ["precision", "recall", "f1score"])


#     print("Summary:")
#     print(eval_df)
#     eval_df.to_csv(os.path.join(args.output_dir, args.model+osp.basename(args.test_csv).split('.')[0]+"_summary.csv"), encoding='utf-8-sig')

    results = {"image_path": test_df["image_path"],
                "class_id": test_df["class_id"],
                "prob": probs, "pred_class": pred_classes}
    all_probs = np.concatenate(all_probs, 0)

#     for n in range(len(id_to_class)):
#         results[id_to_class[n]] = all_probs[:, n]
#     out_df = pd.DataFrame(results)
#     out_df.to_csv(os.path.join(args.output_dir, args.model+osp.basename(args.test_csv).split('.')[0]+"_output.csv"), index=False, encoding='utf-8-sig')
    
    
    with open('.dataset/hierarchy_classification/version3/level_names_dict.pkl','rb') as fo:
        level_names_dict = pickle.load(fo)
    df_species = pd.DataFrame(all_probs, columns=['species_' + x for x in level_names_dict['species']])
    df_test = pd.read_csv(args.test_csv)
    df_res = pd.concat([df_test, df_species], axis = 1)
    df_res.to_csv(osp.join(args.output_dir, os.path.basename(args.test_csv.replace('.','_res.'))), encoding='utf-8-sig',index=False)
    
    
#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     print('Testing time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Testing')

    parser.add_argument('--test_csv', default='dataset/hierarchy_classification/version2023/test_image_path_keep_species_all.csv', help='test csv file')
    parser.add_argument('--model', default='swint', help='model')
    parser.add_argument('--num_classes', default=23, help='total class')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--output_dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='.', help='resume from checkpoint')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
