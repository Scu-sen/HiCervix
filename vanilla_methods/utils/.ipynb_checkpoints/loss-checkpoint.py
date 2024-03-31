# -*- coding: utf-8 -*-
# Copyright (c) 2020, Tencent Inc. All rights reserved.
# Author: huye
# Date: 2020-03-04

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Tuple
from torch import Tensor
import numpy as np
import os.path as osp
import pickle
import numpy as np
from sklearn.metrics import accuracy_score



class CrossEntropyLossV2(nn.CrossEntropyLoss):
    """Cross-entropy loss with label smoothing.
    label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
    meaning the confidence on label values are relaxed.
    e.g. label_smoothing=0.2 means that we will use a value of 0.1
    for label 0 and 0.9 for label 1"""

    def __init__(self, label_smoothing=0.1, weight=None, size_average=None,
            ignore_index=-100, reduce=None, reduction='mean'):
        super(CrossEntropyLossV2, self).__init__(weight, size_average,
                ignore_index, reduce, reduction)
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        num_classes = input.size(1)
        label_neg = self.label_smoothing / num_classes
        label_pos = 1. - label_neg * (num_classes - 1)
        with torch.no_grad():
            ignore = target == self.ignore_index
            n_valid = (ignore == 0).sum()
            onehot_label = torch.empty_like(input).fill_(label_neg).scatter_(
                    1, target.unsqueeze(1), label_pos)

        loss = F.log_softmax(input, dim=1)
        loss = -torch.sum(loss * onehot_label, dim=1) * (1. - ignore.float())
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class CrossEntropyLossV3(nn.CrossEntropyLoss):
    """Cross-entropy loss with label smoothing.
    label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
    meaning the confidence on label values are relaxed.
    e.g. label_smoothing=0.2 means that we will use a value of 0.1
    for label 0 and 0.9 for label 1"""

    def __init__(self, label_smoothing=0.1, weight=None, size_average=None,
            ignore_index=-100, reduce=None, reduction='mean'):
        super(CrossEntropyLossV2, self).__init__(weight, size_average,
                ignore_index, reduce, reduction)
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        loss = F.log_softmax(input, dim=1)
        loss = -torch.sum(loss * target, dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class CrossEntropyLossV4(nn.CrossEntropyLoss):
    """Cross-entropy loss with pair-wise smoothing.
    """
    '''
    def __init__(self, label_smoothing=0.1, weight=None, size_average=None,
            ignore_index=-100, reduce=None, reduction='mean'):
        super(CrossEntropyLossV2, self).__init__(weight, size_average,
                ignore_index, reduce, reduction)
        self.label_smoothing = label_smoothing
    '''
    def forward(self, input, target):
        pair_wise_matrix =torch.tensor([[0.9, 0.1, 0, 0.1],
                            [0.1, 0.8, 0.2, 0.1],
                            [0,  0.1,0.8,0],
                            [0,0,0,0.8]]).cuda()
        #print(input.shape)
        #print(target.shape)
        target = target.unsqueeze(1)
        onehot_label = torch.zeros_like(input).scatter_(1,target,1)
        smooth_label = onehot_label.float()@pair_wise_matrix
        loss = F.log_softmax(input, dim=1)
        loss = -torch.sum(loss * smooth_label, dim=1)
        return loss.mean()
        '''
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
       
        #return F.cross_entropy(input, target)
        loss = F.log_softmax(input, dim=1)
        loss = -torch.sum(loss * target, dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        '''
class CrossEntropyLossV4_plus_binary(nn.CrossEntropyLoss):
    """Cross-entropy loss with pair-wise smoothing.
    """
    def forward(self, input, target):
        pair_wise_matrix =torch.tensor([[0.9, 0.1, 0, 0.1],
                            [0.1, 0.8, 0.2, 0.1],
                            [0,  0.1,0.8,0],
                            [0,0,0,0.8]]).cuda()
        target = target.unsqueeze(1)
        onehot_label = torch.zeros_like(input).scatter_(1,target,1)
        smooth_label = onehot_label.float()@pair_wise_matrix
        #print(smooth_label.shape)
        target_binary = torch.cat([smooth_label[:,0].unsqueeze(1),torch.sum(smooth_label[:,1:],1,keepdim=True)],1)
        #print(input.shape)
        input_binary = torch.cat([input[:,0].unsqueeze(1),torch.sum(input[:,1:],1,keepdim=True)],1)#todo  
        loss_binary = F.log_softmax(input_binary,dim=1)
        loss_binary = -torch.sum(loss_binary*target_binary,dim=1)
        loss = F.log_softmax(input, dim=1)
        loss = -torch.sum(loss * smooth_label, dim=1)
        return loss.mean()+loss_binary.mean()

class CrossEntropyLoss_plus_RegressionLoss(nn.Module):
    
    def forward(self, input, target):

        target = target.unsqueeze(1)
        num_classes = input.size(1)
        regression_multiplier = torch.FloatTensor(np.arange(1, num_classes+1)).cuda()
        onehot_label = torch.zeros_like(input).scatter_(1,target,1)
        regression_input = torch.mm(F.softmax(input, dim=1),regression_multiplier.unsqueeze(1))
        regression_label = target + 1

        y_ = F.log_softmax(input, dim=1)
        ce_loss = -torch.sum(y_ * onehot_label, dim=1)
        regression_loss = torch.pow(regression_input - regression_label,2)
        
#         print(ce_loss.mean())
        #print(regression_loss.mean())
        return ce_loss.mean()+regression_loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()




def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss

class CircleLoss_v2(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss_v2, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, feature:Tensor, label: Tensor) -> Tensor:
        sp, sn = convert_label_to_similarity(F.normalize(feature), label)
        #print(sp.shape)
        #print(sn.shape)
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss

    def convert_label_to_similarity(self, normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
        label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

        positive_matrix = label_matrix.triu(diagonal=1)
        negative_matrix = label_matrix.logical_not().triu(diagonal=1)

        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)
        return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()
    def forward(self, input, weight):
        input = F.softmax(input, dim=1)
        input = torch.clamp(input, min=1e-6)
        loss = -torch.sum(torch.mul(input, torch.log2(input)),dim=1)
        weighted_loss = weight*loss # per sample weight
        #         print(loss.shape)
        #         print(weighted_loss.shape)
        #         print(weighted_loss)
        return weighted_loss.mean()

class CrossEntropyLoss_Prob(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss_Prob, self).__init__()

    def forward(self, cls_probits, label, weight=None, avg_factor=None):
        label = label.unsqueeze(1)
        onehot_label = torch.zeros_like(cls_probits).scatter_(1,label,1)
        cls_probits = torch.clamp(cls_probits, min=1e-6)
        loss = -torch.sum(torch.log(cls_probits) * onehot_label, dim=1)
        #loss = F.log_softmax(cls_probits, dim=1)
        #loss = -torch.sum(loss * onehot_label, dim=1)
        # num_effective_bins = np.sum(weight)
        if weight is not None:
            loss = loss*weight
        if avg_factor is not None:
            loss = loss.sum()/avg_factor
        else:
            loss = loss.mean()
        return loss


class HierarchyLoss(nn.Module):
    def __init__(self,
                 alpha = 1,
                 metadir = '/mnt/group-ai-medical/private/daviddecai/dataset/hierachy_classficattion/hierarchy_metadata'):
        super(HierarchyLoss,self).__init__()
        self.alpha = alpha
        self.hierarchy_label2treelabel = torch.load(osp.join(metadir, 'hierarchy_label2treelabel.pt')).cuda()
        self.hierarchy_binslabel = torch.load(osp.join(metadir, 'hierarchy_binslabel.pt')).cuda()
        self.hierarchy_binsplit = torch.load(osp.join(metadir, 'hierarchy_binsplit.pt')).cuda()
        self.hierarchy_id2id = pickle.load(open(osp.join(metadir, 'hierarchy_id2id.pkl'),'rb'))
        self.entropyloss_id2bins = torch.load(osp.join(metadir,'entropyloss_id2bins.pt')).cuda()
        self.loss_cls = CrossEntropyLoss_Prob()
        self.loss_entropy = EntropyLoss()
    def forward(self, cls_score, labels):
        """cls_score: logits of 12 classes including the ASC class
        """
        new_labels, new_weights, new_avgfactors, entropy_bins_weight = self._remap_labels(labels)
        new_preds = self._slice_preds(cls_score)
        cls_score_probits = self._merge_score_train(cls_score)
        new_preds_probits = self._slice_preds(cls_score_probits)
        num_bins = len(new_labels)
        loss_cls = 0
        loss_entropy = 0
        for i in range(num_bins):
            loss_cls += self.loss_cls(
                    # new_preds[i],
                    new_preds_probits[i],
                    new_labels[i],
                    new_weights[i],
                    new_avgfactors[i])
            loss_entropy += self.loss_entropy(new_preds[i], entropy_bins_weight[i])

        #loss = loss_cls + self.alpha*loss_entropy
        loss = loss_cls - self.alpha*loss_entropy
        loss = loss/num_bins
        return loss
    # @staticmethod
    def accuracy(self,  cls_score, labels):
        cls_score_probits = self._merge_score(cls_score)
        pred_class_ids = []
        for pred_score in cls_score_probits:
            pred_score = pred_score.tolist()
            if pred_score[5] > 0.5:
                pred_class_id = 5
                if np.max(pred_score[6:9]) > 0.5:
                    pred_class_id = np.argmax(pred_score[6:9]) + 6
                    if np.max(pred_score[9:]) > 0.5:
                        pred_class_id = np.argmax(pred_score[9:]) + 9
            else:
                pred_class_id = np.argmax(pred_score[0:5])
            pred_class_ids.append(pred_class_id)
        batch_accuracy = accuracy_score(labels.tolist(), pred_class_ids)
        return batch_accuracy, pred_class_ids

    def _merge_score(self, cls_score):
        '''
        Do softmax in each bin. calculate the conditional probabilities
        from 32 classes to 28+1 classes,
        '''

        new_preds = self._slice_preds(cls_score)
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]
        new_scores = torch.cat(new_scores, dim=1)
        new_scores_cond = torch.zeros_like(new_scores).cuda()
        for i in range(new_scores.shape[1]):
            if i in self.hierarchy_id2id:
                class_new_scores = new_scores[:,self.hierarchy_id2id[i]]
                if class_new_scores.ndim == 1:
                    class_new_scores = class_new_scores.unsqueeze(1)
                new_scores_cond[:,i] = torch.prod(class_new_scores, dim=1)
        fg_merge = new_scores_cond[:,list(self.hierarchy_id2id.keys())]   # the order of classes
        # merge = torch.cat([new_scores[:,0].unsqueeze(1), fg_merge], dim=1)
        return fg_merge

    def _merge_score_train(self, cls_score):
        '''
        Do softmax in each bin. calculate the conditional probabilities
        from 32 classes to 28+1 classes,
        '''

        new_preds = self._slice_preds(cls_score)
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]
        new_scores = torch.cat(new_scores, dim=1)
        # new_scores_cond = torch.zeros_like(new_scores).cuda()
        new_scores_cond = new_scores.clone()
        for i in range(new_scores.shape[1]):
            if i in self.hierarchy_id2id:
                class_new_scores = new_scores[:,self.hierarchy_id2id[i]]
                if class_new_scores.ndim == 1:
                    class_new_scores = class_new_scores.unsqueeze(1)
                new_scores_cond[:,i] = torch.prod(class_new_scores, dim=1)
        return new_scores_cond

    def _remap_labels(self, labels):

        num_bins = self.hierarchy_label2treelabel.shape[0]
        new_labels = []
        new_weights = []
        new_avg = []
        entropy_bins_weight = []
        for i in range(num_bins):
            mapping = self.hierarchy_label2treelabel[i]
            new_bin_label = mapping[labels]
            new_labels.append(new_bin_label)
            #todo 大类有标签才计算小类下的CE loss, set weight as 0
            bins_mapping = self.hierarchy_binslabel[i]
            # bins_weight.append(bins_mapping[labels])
            # weight = torch.ones_like(new_bin_label)
            weight = bins_mapping[labels]
            new_weights.append(weight)
            avg_factor = max(torch.sum(weight).float().item(), 1.)
            new_avg.append(avg_factor)

            entropy_bins_mapping = self.entropyloss_id2bins[i]
            entropy_bins_label = entropy_bins_mapping[labels]
            entropy_bins_weight.append(entropy_bins_label)

        return new_labels, new_weights, new_avg, entropy_bins_weight

    def _slice_preds(self, cls_score):
        new_preds = []

        num_bins = self.hierarchy_binsplit.shape[0]
        for i in range(num_bins):
            start, length = self.hierarchy_binsplit[i][0], self.hierarchy_binsplit[i][1] - self.hierarchy_binsplit[i][0]
            sliced_pred = cls_score.narrow(1, start, length)
            new_preds.append(sliced_pred)
        return new_preds
