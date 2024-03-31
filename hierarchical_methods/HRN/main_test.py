import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms, models
import torch.hub
import argparse
from torch.optim import lr_scheduler
import albumentations as albu
import cv2

from RFM import HIFD2

from tree_loss import TreeLoss
from dataset import CubDataset, CubDataset2, AirDataset, AirDataset2, TCTDataset
from train_test import test, test_AP, test_v1
from train_test import train


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch Deployment')
    parser.add_argument('--worker', default=32, type=int, help='number of workers')
    parser.add_argument('--model', type=str, default='./models_TCT/model_TCT_100_384_p1.0_bz48_ResNet-50__Cos.pth', help='Path of trained model')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--proportion', type=float, default=1.0, help='Proportion of species label')  
    parser.add_argument('--epoch', type=int, default=50,  help='Epochs')
    parser.add_argument('--batch', type=int, default=64, help='batch size')      
    parser.add_argument('--dataset', type=str, default='TCT', help='dataset name')
    parser.add_argument('--img_size', type=str, default='384', help='image size')
    parser.add_argument('--lr_adjt', type=str, default='Cos', help='Learning rate schedual')
    parser.add_argument('--device', nargs='+', default='0', help='GPU IDs for DP training')

    args = parser.parse_args()

    if args.proportion == 0.1: 
        args.epoch = 100
        args.batch = 8
        args.lr_adjt = 'Step'
    
    return args


if __name__ == '__main__':
    args = arg_parse()
    print('==> proportion: ', args.proportion)
    print('==> epoch: ', args.epoch)
    print('==> batch: ', args.batch)
    print('==> dataset: ', args.dataset)
    print('==> img_size: ', args.img_size)
    print('==> device: ', args.device)
    print('==> lr_adjt: ', args.lr_adjt)

    # Hyper-parameters
    nb_epoch = args.epoch
    batch_size = args.batch
    num_workers = args.worker

    # Preprocess
#     transform_train = transforms.Compose([
#     transforms.Resize((550, 550)),
#     transforms.RandomCrop(448, padding=8),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
#     transform_test = transforms.Compose([
#         transforms.Resize((550, 550)),
#         transforms.CenterCrop(448),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.ColorJitter(brightness=10, contrast=10, saturation=20, hue=0.1),
        transforms.ToTensor(),
        normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])
    test_albu_transform = albu.Compose([
        albu.PadIfNeeded(min_height=1000, min_width=1000,
            border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), always_apply=True),
        albu.CenterCrop(700, 700, always_apply=True),
        albu.Resize(384, 384, interpolation=cv2.INTER_LINEAR, always_apply=True),
        ])
    
    os.makedirs('./models_'+args.dataset, exist_ok=True)
    # Data

    if args.dataset == 'TCT':
#         train_list = '../Datasets/CUB-200-2011/train_images_4_level_V1.txt'
#         test_list = '../Datasets/CUB-200-2011/test_images_4_level_V1.txt'
        #test_dir = '/home/datasets/HI_Datasets/CUB2011/CUB_200_2011/test'
        train_csv = 'dataset/hierarchy_classification/version2023/train_image_path_hrn.csv'
#         test_csv = 'dataset/hierarchy_classification/version2023/val_image_path_hrn.csv'
        test_csv = 'dataset/hierarchy_classification/version2023/test_image_path_hrn.csv'
        trees = [[25,  0,  4],
               [26,  0,  5],
               [27,  0,  6],
               [28,  0,  7],
               [29,  0,  8],
               [30,  0,  9],
               [31,  0, 10],
               [32,  0, 11],
               [33,  1, 12],
               [34,  1, 13],
               [35,  1, 14],
               [36,  1, 15],
               [37,  1, 16],
               [38,  2, 17],
               [39,  2, 17],
               [40,  2, 18],
               [41,  2, 19],
               [42,  2, 19],
               [43,  3, 20],
               [44,  3, 21],
               [45,  3, 22],
               [46,  3, 23],
               [47,  3, 24]]
        levels = 3
        total_nodes = 48
        #trainset = TCTDataset(train_csv, transform_train, re_level='family', proportion=args.proportion)
        # Uncomment this line for testing OA results
        testset = TCTDataset(test_csv, test_transform, albu_transform=test_albu_transform, re_level='class', proportion=1.0)
        

    
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    # GPU
    device = torch.device("cuda:" + args.device[0])
    
    # RFM from scrach
    backbone = models.resnet50(pretrained=False)
    backbone.load_state_dict(torch.load('./pre-trained/resnet50-19c8e357.pth'))
    # backbone = models.resnext101_32x8d(pretrained=False)
    # backbone.load_state_dict(torch.load('./pre-trained/resnext101_32x8d-8ba56ff5.pth'))
    net = HIFD2(backbone, 1024, args.dataset)

    #RFM from trained model
#     checkpoint = torch.load(resume)
#     net.load_state_dict(checkpoint['state_dict'])
    net = torch.load(args.model)

    net.to(device)

    # Loss functions
    CELoss = nn.CrossEntropyLoss()
    tree = TreeLoss(trees, total_nodes, levels, device)
    
    if args.proportion > 0.1:       # for p > 0.1
        optimizer = optim.SGD([
            {'params': net.classifier_1.parameters(), 'lr': 0.002},
            {'params': net.classifier_2.parameters(), 'lr': 0.002},
            {'params': net.classifier_3.parameters(), 'lr': 0.002},
            {'params': net.classifier_3_1.parameters(), 'lr': 0.002},
            {'params': net.fc1.parameters(), 'lr': 0.002},
            {'params': net.fc2.parameters(), 'lr': 0.002},
            {'params': net.fc3.parameters(), 'lr': 0.002},
            {'params': net.conv_block1.parameters(), 'lr': 0.002},
            {'params': net.conv_block2.parameters(), 'lr': 0.002},
            {'params': net.conv_block3.parameters(), 'lr': 0.002},
            {'params': net.features.parameters(), 'lr': 0.0002}
        ],
            momentum=0.9, weight_decay=5e-4)
    
    else:     # for p = 0.1
        optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
    
    save_name = args.dataset+'_'+str(args.epoch)+'_'+str(args.img_size)+'_p'+str(args.proportion)+'_bz'+str(args.batch)+'_ResNet-50_'+'_'+args.lr_adjt
#     train(nb_epoch, net, trainloader, testloader, optimizer, scheduler, args.lr_adjt, args.dataset, CELoss, tree, device, args.device, save_name)

    # Evaluate OA
    print("evaluating...")
    test_v1(net, testloader, CELoss, tree, device, args.dataset)

    # Evaluate Average PRC
    # test_AP(net, args.dataset, testset, testloader, device)
