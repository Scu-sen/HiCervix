import argparse
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F

import torchvision.transforms as transforms
import numpy as np
from models import API_Net 
from datasets import RandomDataset, BatchDataset, BalancedBatchSampler
from utils import accuracy, AverageMeter, save_checkpoint
from dataset_tct import InputDataset
import albumentations as albu
import cv2
import time
import pickle
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--exp_name', default=None, type=str,
                    help='name of experiment')
parser.add_argument('--data', metavar='DIR',default='',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--evaluate-freq', default=10, type=int,
                    help='the evaluation frequence')
parser.add_argument('--resume', default='model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--n_classes', default=23, type=int,
                    help='the number of classes')
parser.add_argument('--n_samples', default=4, type=int,
                    help='the number of samples per class')
parser.add_argument('--train_csv', default='dataset/hierarchy_classification/version2023/train_image_path_keep_species.csv')
parser.add_argument('--val_csv', default='dataset/hierarchy_classification/version2023/val_image_path_keep_species.csv')
parser.add_argument('--test_csv', default='dataset/hierarchy_classification/version2023/test_image_path_keep_species_all.csv')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def load_data(train_csv, val_csv):
    # Data loading code
    print("Loading data")
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

    train_albu_transform = albu.Compose([
        albu.PadIfNeeded(min_height=1000, min_width=1000,
            border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), always_apply=True),
        #albu.Rotate(limit=180, interpolation=cv2.INTER_LINEAR,
         #   border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.8),
        #albu.CenterCrop(224, 224, always_apply=True),
        albu.RandomCrop(700, 700, always_apply=True),
        albu.RandomBrightnessContrast(),
#         albu.HueSaturationValue(),
#         albu.OneOf([
#             albu.IAAAdditiveGaussianNoise(p=0.5),
#             albu.GaussNoise(p=0.5),
#             ], p=0.5),
#         albu.OneOf([
#             albu.Blur(blur_limit=7, p=0.5),
#             albu.MedianBlur(blur_limit=7, p=0.5),
#             #albu.MotionBlur(blur_limit=7, p=0.5),
#             ], p=0.5),
        albu.Resize(384, 384, interpolation=cv2.INTER_LINEAR, always_apply=True),
        ])

    test_albu_transform = albu.Compose([
        albu.PadIfNeeded(min_height=1000, min_width=1000,
            border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), always_apply=True),
        albu.CenterCrop(700, 700, always_apply=True),
        albu.Resize(384, 384, interpolation=cv2.INTER_LINEAR, always_apply=True),
        ])

    print("Loading training data")
    st = time.time()
    dataset = InputDataset(train_csv, True, train_transform,
            albu_transform=train_albu_transform)
    print("Took", time.time() - st)

    print("Loading validation data")
    dataset_test = InputDataset(val_csv, False, test_transform,
            albu_transform=test_albu_transform)

    return dataset, dataset_test


best_prec1 = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    global args, best_prec1
    args = parser.parse_args()
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    np.random.seed(2)


    # create model
    model = API_Net()
    model = model.to(device)
    model.conv = nn.DataParallel(model.conv)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_conv = torch.optim.SGD(model.conv.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    fc_parameters = [value for name, value in model.named_parameters() if 'conv' not in name]
    optimizer_fc = torch.optim.SGD(fc_parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.resume:
        if os.path.isfile(args.resume):
            print('loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_conv.load_state_dict(checkpoint['optimizer_conv'])
            optimizer_fc.load_state_dict(checkpoint['optimizer_fc'])
            print('loaded checkpoint {}(epoch {})'.format(args.resume, checkpoint['epoch']))
        else:
            print('no checkpoint found at {}'.format(args.resume))


    cudnn.benchmark = True
    # Data loading code
    train_dataset, val_dataset = load_data(args.train_csv, args.test_csv)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    prec1 = validate(val_loader, model, criterion)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    #softmax_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    species_probs = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            input_var = input.to(device)
            target_var = target.to(device).squeeze()

            # compute output
            logits = model(input_var, targets=None, flag='val')
            #print(logits[0])
            # print(logits.shape)  32x200
            #print(target_var)  32
            #softmax_loss = criterion(logits, target_var)
            #softmax_loss_v2 = criterion(logits[:,:23], target_var)
            #print(softmax_loss, softmax_loss_v2)
            species_probs.extend(F.softmax(logits[:,:23], dim=-1).tolist())

            prec1= accuracy(logits, target_var, 1)
            prec5 = accuracy(logits, target_var, 5)
            #softmax_losses.update(softmax_loss.item(), logits.size(0))
            top1.update(prec1, logits.size(0))
            top5.update(prec5, logits.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()



            if i % args.print_freq == 0:
                print('Time: {time}\nTest: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time,
                        top1=top1, top5=top5, time=time.asctime(time.localtime(time.time()))))
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        with open('dataset/hierarchy_classification/version3/level_names_dict.pkl','rb') as fo:
            level_names_dict = pickle.load(fo)
#         df_order = pd.DataFrame(order_probs, columns=['order_' + x for x in level_names_dict['order']])
#         df_family = pd.DataFrame(family_probs, columns=['family_' + x for x in level_names_dict['family']])
        df_species = pd.DataFrame(species_probs, columns=['species_' + x for x in level_names_dict['species']])
        df_test = pd.read_csv(args.test_csv)
        df_res = pd.concat([df_test, df_species], axis = 1)
        df_res.to_csv(os.path.basename(args.test_csv.replace('.','_res.')), encoding='utf-8-sig',index=False)
        
    return top1.avg




if __name__ == '__main__':
    main()
