from __future__ import print_function
import os
import cv2
import time
import shutil
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import albumentations as albu
import pickle
import pandas as pd
import os.path as osp
from dataset import InputDataset
# from torchstat import stat

from tct_get_tree_target import *


logger = logging.getLogger('fine-grained-or-not')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

BATCH_SIZE = 32
batch_size = BATCH_SIZE
Hiden_Number = 512
lr = 0.1
nb_epoch = 100 #50

train_csv = 'dataset/hierarchy_classification/version2023/train_image_path.csv'
# val_csv = 'dataset/hierarchy_classification/version2023/val_image_path.csv'
val_csv = 'dataset/hierarchy_classification/version2023/test_image_path.csv'
save = 'classification_hierarchy/vanilla_single/output_0512_384'
resume = 'output_0512_384/model_best.pth.tar'

is_test = True

if not os.path.isdir(save):
    os.makedirs(save)

criterion = nn.CrossEntropyLoss(ignore_index=-1)
criterion_NLLLoss = nn.NLLLoss()


#Data
print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.RandomCrop(224, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])

# transform_test = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])

def load_data(train_csv, val_csv, distributed):
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

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


dataset, dataset_test, train_sampler, test_sampler = load_data(train_csv, val_csv, False)
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size,
    sampler=train_sampler, num_workers=16, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size,
    sampler=test_sampler, num_workers=16, pin_memory=True)

# train_data = torchvision.datasets.ImageFolder(root+'/dataset/train', transform=transform_train)
# test_data = torchvision.datasets.ImageFolder(root+'/dataset/test', transform=transform_test)
# # train_data = cub200(root, train=True, transform=transform_train)
# # test_data = cub200(root, train=False, transform=transform_test)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, 
#                     shuffle=True, num_workers=16, pin_memory=True)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, 
#                     shuffle=False, num_workers=16, pin_memory=True)

class model_bn(nn.Module):
    def __init__(self, model, hid_dim=512, label_hierarchy=[4, 21, 23]):  #[13, 38, 200]

        super(model_bn, self).__init__() 

        self.features_2 =  nn.Sequential(*list(model.children())[:-2])

#         self.max = nn.MaxPool2d(kernel_size=7, stride=7)
        self.max = nn.AdaptiveAvgPool2d((1,1))

        self.feature_dim = 2048 * 1 * 1
        self.split_dim = hid_dim // len(label_hierarchy)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            #nn.Dropout(0.5),
            nn.Linear(self.feature_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            #nn.Linear(feature_size, classes_num),
        )
        
        self.classifiers = nn.ModuleList()
        for i in range(len(label_hierarchy)):
            self.classifiers.append(nn.Linear(hid_dim, label_hierarchy[i]))
 
    def forward(self, x, targets):
        x = self.features_2(x)   
#         print(targets)

        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) # N * 512
        
        logits = []
        for i, classifier in enumerate(self.classifiers):
            logits.append(classifier(x))
        
        order_out = F.softmax(logits[0], dim=-1)
        family_out = F.softmax(logits[1], dim=-1)
        species_out = F.softmax(logits[2], dim=-1)
        
        order_targets, family_targets, species_target= get_order_family_target(targets)
#         print(order_targets)
#         print(family_targets)
#         ce_loss_order = criterion_NLLLoss(torch.log(order_out), order_targets) # 13
#         ce_loss_family = criterion_NLLLoss(torch.log(family_out), family_targets) # 38
#         ce_loss_species = criterion_NLLLoss(torch.log(species_out), targets)
        ce_loss_order = criterion(order_out, order_targets) # 13
        ce_loss_family = criterion(family_out, family_targets) # 38
#         ce_loss_species = criterion(species_out, targets)
        ce_loss_species = criterion(species_out, species_target)
        
        ce_loss =  ce_loss_order + ce_loss_family + ce_loss_species 

#         return ce_loss, [species_out,targets], [family_out, family_targets],\
#                         [order_out, order_targets]
        return ce_loss, [species_out, species_target], [family_out, family_targets],\
                        [order_out, order_targets]

                    
def train(epoch, net, trainloader, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0

    order_correct = 0
    family_correct = 0
    species_correct = 0
    
    order_total = 0
    family_total= 0
    species_total= 0

    idx = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        idx = batch_idx

        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        #out, ce_loss = net(inputs, targets)

        ce_loss,\
        [species_out, species_targets],\
        [family_out, family_targets],\
        [order_out, order_targets] = net(inputs, targets)

        loss = ce_loss
#         print(loss.item())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        _, order_predicted = torch.max(order_out.data, 1)
        order_total += order_targets.size(0)
        order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

        _, family_predicted = torch.max(family_out.data, 1)
        family_total += family_targets.size(0)
        family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()

        _, species_predicted = torch.max(species_out.data, 1)
        species_total += species_targets.size(0)
        species_correct += species_predicted.eq(species_targets.data).cpu().sum().item()
    
    train_order_acc = 100.*order_correct/order_total
    train_family_acc = 100.*family_correct/family_total
    train_species_acc = 100.*species_correct/species_total

    train_loss = train_loss/(idx+1) 
    print('Iteration %d, train_order_acc = %.5f,train_family_acc = %.5f,\
          train_species_acc = %.5f, train_loss = %.6f' % \
              (epoch, train_order_acc,train_family_acc,train_species_acc,train_loss))
    return train_order_acc, train_family_acc,train_species_acc,train_loss


def test(epoch, net, testloader, optimizer):
    net.eval()
    test_loss = 0
    
    order_correct = 0
    family_correct = 0
    species_correct = 0

    order_total = 0
    family_total= 0
    species_total= 0
    
    order_probs = []
    family_probs = []
    species_probs = []
    
    idx = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        with torch.no_grad():
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            #out, ce_loss = net(inputs,targets)

            ce_loss,\
            [species_out, species_targets],\
            [family_out, family_targets],\
            [order_out, order_targets] = net(inputs, targets)

            test_loss += ce_loss.item()

            _, order_predicted = torch.max(order_out.data, 1)
            order_total += order_targets.size(0)
            order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

            _, family_predicted = torch.max(family_out.data, 1)
            family_total += family_targets.size(0)
            family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()

            _, species_predicted = torch.max(species_out.data, 1)
            species_total += species_targets.size(0)
            species_correct += species_predicted.eq(species_targets.data).cpu().sum().item()
            
            order_probs.extend(order_out.tolist())
            family_probs.extend(family_out.tolist())
            species_probs.extend(species_out.tolist())
            

    test_order_acc = 100.*order_correct/order_total
    test_family_acc = 100.*family_correct/family_total
    test_species_acc = 100.*species_correct/species_total

    test_loss = test_loss/(idx+1)
    print('Iteration %d, test_order_acc = %.5f,test_family_acc = %.5f,\
          test_species_acc = %.5f, test_loss = %.6f' % \
              (epoch, test_order_acc,test_family_acc,test_species_acc,test_loss))
    return test_order_acc, test_family_acc,test_species_acc, (order_probs, family_probs, species_probs)


def cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % (nb_epoch  ))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch )
    cos_out = np.cos(cos_inner) + 1
    return float( 0.1 / 2 * cos_out)

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filename = os.path.join(checkpoint, filename)
    torch.save(state, filename)
    if is_best:
        print("\tSaving new best model")
        shutil.copyfile(filename, os.path.join(checkpoint, 'model_best.pth.tar'))


use_cuda = torch.cuda.is_available()
print('==> Building model..')
net =models.resnet50(pretrained=True)
net =model_bn(net, Hiden_Number)
# stat(net, (3, 448, 448))

if use_cuda:
    net.classifiers.cuda()

    net.fc.cuda()
    net.features_2.cuda()

    cudnn.benchmark = True

optimizer = optim.SGD(
            [{'params': net.classifiers.parameters(), 'lr': 0.1},
             {'params': net.fc.parameters(),   'lr': 0.1},
             {'params': net.features_2.parameters(),   'lr': 0.01}], 
            momentum=0.9, weight_decay=5e-4)


if __name__ == '__main__':
#     try:
        # main(params)
    if is_test:
        print(is_test)
        epoch = 20
        if resume:
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                net.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}'"
                      .format(resume))
            else:
                print("=> no checkpoint found at '{}'".format(resume))
                exit()
        print('Evaluating...')
        test_order_acc, test_family_acc, test_species_acc, output_probs = test(epoch, net, test_loader, optimizer)
        order_probs, family_probs, species_probs = output_probs
        with open('dataset/hierarchy_classification/version3/level_names_dict.pkl','rb') as fo:
            level_names_dict = pickle.load(fo)
        df_order = pd.DataFrame(order_probs, columns=['order_' + x for x in level_names_dict['order']])
        df_family = pd.DataFrame(family_probs, columns=['family_' + x for x in level_names_dict['family']])
        df_species = pd.DataFrame(species_probs, columns=['species_' + x for x in level_names_dict['species']])
        df_val = pd.read_csv(val_csv)
        df_res = pd.concat([df_val, df_order, df_family, df_species], axis = 1)
        df_res.to_csv(osp.join(osp.dirname(resume), osp.basename(val_csv).replace('.','_res.')), encoding='utf-8-sig',index=False)
        
        
    else:
        max_val_acc = 0
        for epoch in range(nb_epoch):
            optimizer.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch) 
            optimizer.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch) 
            optimizer.param_groups[2]['lr'] =  cosine_anneal_schedule(epoch) / 10

            if resume:
                if os.path.isfile(resume):
                    print("=> loading checkpoint '{}'".format(resume))
                    checkpoint = torch.load(resume)
                    net.load_state_dict(checkpoint['state_dict'])
                    print("=> loaded checkpoint '{}'"
                          .format(resume))
                else:
                    print("=> no checkpoint found at '{}'".format(resume))
                    exit()

            train(epoch, net, train_loader, optimizer)
            test_order_acc, test_family_acc, test_species_acc, _ = test(epoch, net, test_loader, optimizer)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_mcr': max_val_acc,
                'optimizer': optimizer.state_dict(),
            }, test_species_acc > max_val_acc, save) 

            if test_species_acc > max_val_acc:
                max_val_acc = test_species_acc
            print("max_val_acc ==", max_val_acc)
#     except Exception as exception:
#         logger.exception(exception)
#         raise
