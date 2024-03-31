import numpy as np
import torch
from torch.autograd import Variable

# level_1, level_2, level_3 
# 26个类别，但level_3只有23个类别
trees = [[0, 0, 0],
 [0, 1, 1],
 [0, 2, 2],
 [0, 3, 3],
 [0, 4, 4],
 [0, 5, 5],
 [0, 6, 6],
 [0, 7, 7],
 [1, 8, 8],
 [1, 9, 9],
 [1, 10, 10],
 [1, 11, 11],
 [1, 12, 12],
 [2, -1, -1],
 [2, 13, -1],
 [2, 14, 13],
 [2, 15, -1],
 [2, 13, 14],
 [2, 13, 15],
 [2, 15, 16],
 [2, 15, 17],
 [3, 16, 18],
 [3, 17, 19],
 [3, 18, 20],
 [3, 19, 21],
 [3, 20, 22]]

def get_order_family_target(targets):
    order_target_list = []
    family_target_list = []
    species_target_list = []

    for i in range(targets.size(0)):

        order_target_list.append(trees[targets[i]][0])
        family_target_list.append(trees[targets[i]][1])
        species_target_list.append(trees[targets[i]][2])
    
    order_target_list = Variable(torch.from_numpy(np.array(order_target_list)).cuda())   
    family_target_list = Variable(torch.from_numpy(np.array(family_target_list)).cuda())
    species_target_list = Variable(torch.from_numpy(np.array(species_target_list)).cuda())   

    return order_target_list, family_target_list, species_target_list


