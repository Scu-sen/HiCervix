import torch
from torch.nn.modules.activation import Softmax
from utils import *
import os.path as osp
import pandas as pd
import pickle
import copy
import time
from sklearn.metrics import confusion_matrix, average_precision_score


def train(epoches, net, trainloader, testloader, optimizer, scheduler, lr_adjt, dataset, CELoss, tree, device, devices, save_name):
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    max_val_acc = 0
    best_epoch = 0
    if len(devices) > 1:
        ids = list(map(int, devices))
        netp = torch.nn.DataParallel(net, device_ids=ids)
    for epoch in range(epoches):
        epoch_start = time.time()
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0

        order_correct = 0
        family_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0
        family_correct_sig = 0

        order_total = 0
        family_total= 0
        species_total= 0

        idx = 0
        if lr_adjt == 'Cos':
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, epoches, lr[nlr])
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx

            inputs, targets = inputs.to(device), targets.to(device)
            order_targets, family_targets, target_list_sig = get_order_family_target(targets, device, dataset)

            optimizer.zero_grad()

            if len(devices) > 1:
                xc1_sig, xc2_sig, xc3, xc3_sig = netp(inputs)
            else:
                xc1_sig, xc2_sig, xc3, xc3_sig = net(inputs)
            tree_loss = tree(torch.cat([xc1_sig, xc2_sig, xc3_sig], 1), target_list_sig, device)
            if dataset == 'CUB':
                leaf_labels = torch.nonzero(targets > 50, as_tuple=False)
            elif dataset == 'Air':
                leaf_labels = torch.nonzero(targets > 99, as_tuple=False)
            elif dataset == 'TCT':
                leaf_labels = torch.nonzero(targets > 24, as_tuple=False)
                family_labels_1 = torch.nonzero( targets > 24, as_tuple=False)
                family_labels_2 = torch.nonzero(torch.logical_and(targets > 3, targets < 25), as_tuple=False)
#                 print(targets)
#                 print(family_labels.shape)
#                 print(family_labels)
            if leaf_labels.shape[0] > 0:
                if dataset == 'CUB':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 51
                elif dataset == 'Air':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 100
                elif dataset == 'TCT':
                    select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 25
                select_fc_soft = torch.index_select(xc3, 0, leaf_labels.squeeze())
                ce_loss_species = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)
                loss = ce_loss_species + tree_loss
            else:
                loss = tree_loss
            
            if family_labels_1.shape[0] > 0:
                select_family_labels_1 = torch.index_select(family_targets, 0, family_labels_1.squeeze()) 
            if family_labels_2.shape[0] > 0:
                select_family_labels_2 = torch.index_select(targets, 0, family_labels_2.squeeze()) - 4

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
    
            with torch.no_grad():
                _, order_predicted = torch.max(xc1_sig.data, 1)
                order_total += order_targets.size(0)
#                 print(order_targets.size(0))
#                 print(order_predicted.shape)
                order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()
                # todo
#                 _, family_predicted = torch.max(xc2_sig.data, 1)
#                 family_total += family_targets.size(0)
#                 family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()
                if family_labels_1.shape[0]>0:
                    select_xc2_sig = torch.index_select(xc2_sig, 0, family_labels_1.squeeze())
                    _, family_predicted_sig = torch.max(select_xc2_sig.data, 1)
                
                    family_total += select_family_labels_1.size(0)
                    family_correct_sig += family_predicted_sig.eq(select_family_labels_1.data).cpu().sum().item()
                
                if family_labels_2.shape[0]>0:
                    select_xc2_sig = torch.index_select(xc2_sig, 0, family_labels_2.squeeze())
                    _, family_predicted_sig = torch.max(select_xc2_sig.data, 1)
                
                    family_total += select_family_labels_2.size(0)
                    family_correct_sig += family_predicted_sig.eq(select_family_labels_2.data).cpu().sum().item()
                    
#                     print(select_family_labels)
#                     print(family_predicted_sig)
                if leaf_labels.shape[0] > 0:
                    select_xc3 = torch.index_select(xc3, 0, leaf_labels.squeeze())
                    select_xc3_sig = torch.index_select(xc3_sig, 0, leaf_labels.squeeze())
                    _, species_predicted_soft = torch.max(select_xc3.data, 1)
                    _, species_predicted_sig = torch.max(select_xc3_sig.data, 1)
                    species_total += select_leaf_labels.size(0)
                    species_correct_soft += species_predicted_soft.eq(select_leaf_labels.data).cpu().sum().item()
                    species_correct_sig += species_predicted_sig.eq(select_leaf_labels.data).cpu().sum().item()
        
        if lr_adjt == 'Step':
            scheduler.step()

        train_order_acc = 100.*order_correct/order_total
        #train_family_acc = 100.*family_correct/family_total
        train_family_acc = 100.*family_correct_sig/family_total
        train_species_acc_soft = 100.*species_correct_soft/species_total
        train_species_acc_sig = 100.*species_correct_sig/species_total
        train_loss = train_loss/(idx+1)
        epoch_end = time.time()
        print('Iteration %d, train_order_acc = %.5f,train_family_acc = %.5f,train_species_acc_soft = %.5f,train_species_acc_sig = %.5f, train_loss = %.6f, Time = %.1fs' % \
            (epoch, train_order_acc, train_family_acc, train_species_acc_soft, train_species_acc_sig, train_loss, (epoch_end - epoch_start)))

        test_order_acc, test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss = test(net, testloader, CELoss, tree, device, dataset)
        
        if test_species_acc_soft > max_val_acc:
            max_val_acc = test_species_acc_soft
            best_epoch = epoch
            net.cpu()
            torch.save(net, './models_'+dataset+'/model_'+save_name+'.pth')
            net.to(device)

    print('\n\nBest Epoch: %d, Best Results: %.5f' % (best_epoch, max_val_acc))


def test(net, testloader, CELoss, tree, device, dataset):
    epoch_start = time.time()
    with torch.no_grad():
        net.eval()
        test_loss = 0

        order_correct = 0
        family_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0

        order_total = 0
        family_total= 0
        species_total= 0

        idx = 0
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            idx = batch_idx

            inputs, targets = inputs.to(device), targets.to(device)
            order_targets, family_targets, target_list_sig = get_order_family_target(targets, device, dataset)

            xc1_sig, xc2_sig, xc3, xc3_sig = net(inputs)
            tree_loss = tree(torch.cat([xc1_sig, xc2_sig, xc3_sig], 1), target_list_sig, device)
            if dataset == 'CUB':
                leaf_labels = torch.nonzero(targets > 50, as_tuple=False)
                select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 51
            elif dataset == 'Air':
                leaf_labels = torch.nonzero(targets > 99, as_tuple=False)
                select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 100
            elif dataset == 'TCT':
                leaf_labels = torch.nonzero(targets > 24, as_tuple=False)
                family_labels = torch.nonzero(targets > 3, as_tuple=False)
                select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 25
            select_fc_soft = torch.index_select(xc3, 0, leaf_labels.squeeze())
            ce_loss_species = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)
            loss = ce_loss_species + tree_loss

            test_loss += loss.item()
    
            _, order_predicted = torch.max(xc1_sig.data, 1)
            order_total += order_targets.size(0)
            order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

#             _, family_predicted = torch.max(xc2_sig.data, 1)
#             family_total += family_targets.size(0)
#             family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()
            if leaf_labels.shape[0] > 0:
                select_xc3 = torch.index_select(xc3, 0, leaf_labels.squeeze())
                select_xc3_sig = torch.index_select(xc3_sig, 0, leaf_labels.squeeze())
                _, species_predicted_soft = torch.max(select_xc3.data, 1)
                _, species_predicted_sig = torch.max(select_xc3_sig.data, 1)
                species_total += select_leaf_labels.size(0)
                species_correct_soft += species_predicted_soft.eq(select_leaf_labels.data).cpu().sum().item()
                species_correct_sig += species_predicted_sig.eq(select_leaf_labels.data).cpu().sum().item()
#             _, species_predicted_soft = torch.max(xc3.data, 1)
#             _, species_predicted_sig = torch.max(xc3_sig.data, 1)
#             species_total += select_leaf_labels.size(0)
#             species_correct_soft += species_predicted_soft.eq(select_leaf_labels.data).cpu().sum().item()
#             species_correct_sig += species_predicted_sig.eq(select_leaf_labels.data).cpu().sum().item()


        test_order_acc = 100.* order_correct/order_total
#         test_family_acc = 100.* family_correct/family_total
        test_family_acc = 0
        test_species_acc_soft = 100.* species_correct_soft/species_total
        test_species_acc_sig = 100.* species_correct_sig/species_total
        test_loss = test_loss/(idx+1)
        epoch_end = time.time()
        print('test_order_acc = %.5f,test_family_acc = %.5f,test_species_acc_soft = %.5f,test_species_acc_sig = %.5f, test_loss = %.6f, Time = %.4s' % \
             (test_order_acc, test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss, epoch_end - epoch_start))

    return test_order_acc, test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss


def test_v1(net, testloader, CELoss, tree, device, dataset):
    epoch_start = time.time()
    with torch.no_grad():
        net.eval()
        test_loss = 0

        order_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0
        family_correct_sig = 0

        order_total = 0
        family_total= 0
        species_total= 0
        order_probs, family_probs, species_probs = [], [], []
        idx = 0
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            idx = batch_idx

            inputs, targets = inputs.to(device), targets.to(device)
            order_targets, family_targets, target_list_sig = get_order_family_target(targets, device, dataset)

            xc1_sig, xc2_sig, xc3, xc3_sig = net(inputs)
            order_probs.extend(xc1_sig.tolist())
            family_probs.extend(xc2_sig.tolist())
            species_probs.extend(xc3.tolist())
            tree_loss = tree(torch.cat([xc1_sig, xc2_sig, xc3_sig], 1), target_list_sig, device)
            if dataset == 'CUB':
                leaf_labels = torch.nonzero(targets > 50, as_tuple=False)
                select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 51
            elif dataset == 'Air':
                leaf_labels = torch.nonzero(targets > 99, as_tuple=False)
                select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 100
            elif dataset == 'TCT':
                leaf_labels = torch.nonzero(targets > 24, as_tuple=False)
                family_labels_1 = torch.nonzero( targets > 24, as_tuple=False)
                family_labels_2 = torch.nonzero(torch.logical_and(targets > 3, targets < 25), as_tuple=False)
                select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 25
            select_fc_soft = torch.index_select(xc3, 0, leaf_labels.squeeze())
            ce_loss_species = CELoss(select_fc_soft.to(torch.float64), select_leaf_labels)
            loss = ce_loss_species + tree_loss
            
            if family_labels_1.shape[0] > 0:
                select_family_labels_1 = torch.index_select(family_targets, 0, family_labels_1.squeeze()) 
            if family_labels_2.shape[0] > 0:
                select_family_labels_2 = torch.index_select(targets, 0, family_labels_2.squeeze()) - 4
            
            test_loss += loss.item()
    
            _, order_predicted = torch.max(xc1_sig.data, 1)
            order_total += order_targets.size(0)
            order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()
#             print('*'*20)
#             print(xc1_sig.shape)
#             print('order_predicted', order_predicted)
#             print('order_targets', order_targets.data)

            if family_labels_1.shape[0]>0:
                select_xc2_sig = torch.index_select(xc2_sig, 0, family_labels_1.squeeze())
                _, family_predicted_sig = torch.max(select_xc2_sig.data, 1)

                family_total += select_family_labels_1.size(0)
                family_correct_sig += family_predicted_sig.eq(select_family_labels_1.data).cpu().sum().item()

            if family_labels_2.shape[0]>0:
                select_xc2_sig = torch.index_select(xc2_sig, 0, family_labels_2.squeeze())
                _, family_predicted_sig = torch.max(select_xc2_sig.data, 1)

                family_total += select_family_labels_2.size(0)
                family_correct_sig += family_predicted_sig.eq(select_family_labels_2.data).cpu().sum().item()

            if leaf_labels.shape[0] > 0:
                select_xc3 = torch.index_select(xc3, 0, leaf_labels.squeeze())
                select_xc3_sig = torch.index_select(xc3_sig, 0, leaf_labels.squeeze())
                _, species_predicted_soft = torch.max(select_xc3.data, 1)
                _, species_predicted_sig = torch.max(select_xc3_sig.data, 1)
                species_total += select_leaf_labels.size(0)
                species_correct_soft += species_predicted_soft.eq(select_leaf_labels.data).cpu().sum().item()
                species_correct_sig += species_predicted_sig.eq(select_leaf_labels.data).cpu().sum().item()

        test_order_acc = 100.* order_correct/order_total
        test_family_acc = 100.* family_correct_sig/family_total
        test_species_acc_soft = 100.* species_correct_soft/species_total
        test_species_acc_sig = 100.* species_correct_sig/species_total
        test_loss = test_loss/(idx+1)
        epoch_end = time.time()
        
        with open('dataset/hierarchy_classification/version3/level_names_dict.pkl','rb') as fo:
            level_names_dict = pickle.load(fo)
        df_order = pd.DataFrame(order_probs, columns=['order_' + x for x in level_names_dict['order']])
        df_family = pd.DataFrame(family_probs, columns=['family_' + x for x in level_names_dict['family']])
        df_species = pd.DataFrame(species_probs, columns=['species_' + x for x in level_names_dict['species']])
        test_csv = 'dataset/hierarchy_classification/version2023/test_image_path_hrn.csv'
        df_test = pd.read_csv(test_csv)
        df_res = pd.concat([df_test, df_order, df_family, df_species], axis = 1)
        df_res.to_csv(osp.basename(test_csv).replace('.','_res.'), encoding='utf-8-sig',index=False)
        print('test_order_acc = %.5f,test_family_acc = %.5f,test_species_acc_soft = %.5f,test_species_acc_sig = %.5f, test_loss = %.6f, Time = %.4s' % \
             (test_order_acc, test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss, epoch_end - epoch_start))

#     return test_order_acc, test_family_acc, test_species_acc_soft, test_species_acc_sig, test_loss


def test_AP(model, dataset, test_set, test_data_loader, device):
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        model.eval()
        for j, (images, labels) in enumerate(test_data_loader):
            images = images.to(device)
            labels = labels.to(device)
            select_labels = labels[:, test_set.to_eval]
            if dataset == 'CUB' or dataset == 'Air':
                y_order_sig, y_family_sig, y_species_sof, y_species_sig = model(images)
                batch_pMargin = torch.cat([y_order_sig, y_family_sig, torch.softmax(y_species_sof, dim=1)], dim=1).data
            else:
                y_order_sig, y_species_sof, y_species_sig = model(images)
                batch_pMargin = torch.cat([y_order_sig, torch.softmax(y_species_sof, dim=1)], dim=1).data
            
            predicted = batch_pMargin > 0.5
            total += select_labels.size(0) * select_labels.size(1)
            correct += (predicted.to(torch.float64) == select_labels).sum()
            cpu_batch_pMargin = batch_pMargin.to('cpu')
            y = select_labels.to('cpu')
            if j == 0:
                test = cpu_batch_pMargin
                test_y = y
            else:
                test = torch.cat((test, cpu_batch_pMargin), dim=0)
                test_y = torch.cat((test_y, y), dim=0)
        score = average_precision_score(test_y, test, average='micro')
        print('Accuracy:' + str(float(correct) / float(total)))
        print('Precision score:' + str(score))