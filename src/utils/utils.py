import torch
import sys
import ipdb
import os
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from collections import Counter

def save_model(model, optimizer, checkpoints_dir, suff=''):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    torch.save(model.state_dict(), os.path.join(
        checkpoints_dir, 'model' + suff + '.ckpt'))

    torch.save(optimizer.state_dict(), os.path.join(
        checkpoints_dir, 'optim' + suff + '.ckpt'))




class LossChecker:
    def __init__(self, num_losses):
        self.num_losses = num_losses

        self.losses = [ [] for _ in range(self.num_losses) ]

    def update(self, *loss_vals):
        assert len(loss_vals) == self.num_losses

        for i, loss_val in enumerate(loss_vals):
            self.losses[i].append(loss_val)

    def mean(self, last=0):
        mean_losses = [ 0. for _ in range(self.num_losses) ]
        for i, loss in enumerate(self.losses):
            _loss = loss[-last:]
            mean_losses[i] = sum(_loss) / len(_loss)
        return mean_losses




def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def test_acc(x, target_class, is_test_arri, counter_gen, counter_cov, counter_true):
    device_tmp = torch.device('cpu')
    possi_weight_gen = x
    possi_weight_cov = x * is_test_arri


    y_list = target_class.to(device_tmp).numpy().tolist()
    y_list = list(np.array(y_list).ravel())
    counter_true.update(Counter(y_list))

    _, top_1 = torch.topk(possi_weight_gen, 1)
    # ipdb.set_trace()
    top_1 = top_1.squeeze(1)
    x_list_gen = top_1.to(device_tmp).numpy().tolist()
    x_list_gen = list(np.array(x_list_gen).ravel())

    _, top_1 = torch.topk(possi_weight_cov, 1)
    # ipdb.set_trace()
    top_1 = top_1.squeeze(1)
    acc_cov = torch.sum(top_1==target_class)/top_1.shape[0]
    x_list_cov = top_1.to(device_tmp).numpy().tolist()
    x_list_cov = list(np.array(x_list_cov).ravel())

    result_gen = []
    result_cov = []
    for (x_gen, x_cov, y) in zip(x_list_gen, x_list_cov,y_list):
        if (x_gen==y):
            result_gen.append(x_gen)
        if (x_cov==y):
            result_cov.append(x_cov)
    counter_gen.update(Counter(result_gen))
    counter_cov.update(Counter(result_cov))

    true_rate_gen = 0
    for k,v in counter_true.items():
        tmp_rate = counter_gen[k] / v
        true_rate_gen = true_rate_gen + tmp_rate
    true_rate_gen = true_rate_gen / len(counter_true)

    true_rate_cov = 0
    for k, v in counter_true.items():
        tmp_rate = counter_cov[k] / v
        true_rate_cov = true_rate_cov + tmp_rate
    true_rate_cov = true_rate_cov / len(counter_true)

    return true_rate_gen,true_rate_cov,acc_cov



def QFSLloss( x, target_num, is_train_x, is_test_class, args):
    
    possi_weight = x

    is_train_x=is_train_x.to(torch.float)
    
    '''
    Calculate the cross-entropy loss of the source class in the output part
    '''
    target_onehot = F.one_hot(target_num,x.shape[1])
    log_pre = torch.log(possi_weight+1e-7)
    mul_pre = log_pre * target_onehot
    sum_loss = torch.sum(mul_pre,dim=1)
#     loss1 = -torch.sum(sum_loss) / x.shape[0]
    train_loss = sum_loss * is_train_x
    loss1 = -torch.sum(train_loss) / torch.sum(is_train_x)
    

    '''
    Calculate addtional bias loss 
    is_test_class determines which class belongs to the Target class 
    '''

    
    target_class_weight = torch.sum(possi_weight * is_test_class,dim=1)
    additional_bias_loss = torch.log(target_class_weight+(1e-7)) * (1-is_train_x)
    loss2 = - torch.sum(additional_bias_loss) / (torch.tensor(x.shape[0]) - torch.sum(is_train_x) + 1e-7)

    """QFSLloss"""
    loss_all =  loss1 + loss2 * args.bias_weight


    """Accuracy of training set"""
    _,top_1 = torch.topk(possi_weight, 1)
    top_1 = top_1.squeeze(1)
    #ipdb.set_trace()
    result = top_1==target_num
    result_train = result * is_train_x
    true_num_train = torch.sum(result_train)
    true_rate_train = true_num_train / torch.sum(is_train_x)
    loss = {
        "total" : loss_all,
        "cross_loss" : loss1,
        "additional_bias_loss" : loss2,
      
    }
    

    return loss, true_rate_train


