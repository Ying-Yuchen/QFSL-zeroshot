import torch
import torchvision
import sys
import os
from utils.config import Config as C
from torch.nn import functional as F
from torch import nn
from utils.utils import set_random_seed, LossChecker, test_acc, QFSLloss,save_model
from models.QFSLnet import QFSLnet
from data_loader import Corpus
from tqdm import tqdm
import ipdb
from collections import Counter
from args import get_parser
import json



def build_loader(img_path, source_classes_list, target_classes_list, data_info, batch_size,num_wokers):
    corpus = Corpus(img_path, source_classes_list, target_classes_list, data_info, batch_size,num_wokers)
    return corpus.train_dataloader,corpus.test_dataloader


def train_batch(net, img, target_num, is_train, is_test_attr, loss, args, trainer, device, is_train_net):
    X = img.to(device)
    is_train = is_train.to(device)
    is_test_attr = is_test_attr.to(device)
    target_num = target_num.to(device)
    net.train()
    trainer.zero_grad()
    pred = net(X, is_train_net)
    loss_,train_acc_rate = loss( pred,  target_num, is_train, is_test_attr, args)
    l = loss_["total"]
    l.backward()
    trainer.step()
    return loss_, train_acc_rate

'''
net : model
train_iter: return (image : 224*224,
                     attri : Attribute values of the target class , 
                     obj2one_hot : Number of the target class, 
                     is_train : Determine if it is a source class(Source class and target class enter training together during training) )
test_iter:
loss : from utils, QFSLloss
trainer : sgd
num_epochs : C.Model.epoch
is_test_class : a list , Mark each class(in order corresponding to the number of each class) as a source class or not
class_num : Number of classes
add_weight : the weight of addtional bias loss 
device : cuda
'''

def train_model(args,model,train_iter,test_iter,loss,optimizer,is_test_class,device):

    
    test_mca_best = 0.0
    net = model.to(device)
    start = 0
    if args.resume==True:
        start = args.resume_epoch_nums
    if not os.path.exists(C.score_dic_path):
        os.makedirs(C.score_dic_path)
    for epoch in range(start, args.num_epochs):
        net.train()
        loss_checker = LossChecker(4)
        pgbar = tqdm(train_iter)
        for  batch in pgbar:
            is_train_net = True
            (image, target_num, is_train, class_attr) = batch
            l, acc = train_batch( net, image, target_num, is_train, is_test_class,
                                loss,  args, optimizer, device, is_train_net)
            loss_checker.update(l["total"].item(), l["cross_loss"].item(), l["additional_bias_loss"].item(), acc.item())
            pgbar.set_description(
                "[Epoch train #{0}]lr:{6:5f} loss: {2:.5f} = CE {3:.5f} + AB {1} * {4:.5f}   acc={5:.5f}".format(
                    epoch, args.bias_weight,  *loss_checker.mean(0),args.learning_rate))
        with open(C.train_score_path, "a") as fout:
            fout.write(
                "[Epoch train #{0}]lr:{6:5f} loss: {2:.5f} = CE {3:.5f} + AB {1} * {4:.5f}   acc={5:.5f}\n".format(
             epoch, args.bias_weight,  *loss_checker.mean(0),args.learning_rate))
        save_model(net,optimizer,C.checkpoints_dir)

        loss_checker = LossChecker(1)
        counter_gen = Counter()
        counter_cov = Counter()
        counter_true = Counter()
        test_pgbar = tqdm(test_iter)
        net.eval()
        for batch in test_pgbar:
            is_train_net = False
            (image, target_num, is_train, class_attr) = batch
            X = image.to(device)
            target_num = target_num.to(device)
            is_test_class = is_test_class.to(device)
            pred = net(X, is_train_net)
            mca_gen, mca_cov, acc_cov = test_acc(pred,target_num,is_test_class, counter_gen, counter_cov,counter_true)
            loss_checker.update(acc_cov.item())
            test_pgbar.set_description(
                "[Epoch test#{0}]lr={4:5f} mca_gen={1:.5f} mca_cov={2:.5f} acc_cov={3:5f}"
                    .format(epoch, mca_gen, mca_cov, *loss_checker.mean(0),args.learning_rate)
            )
        
        with open(C.test_score_path, "a") as fout:
            fout.write( "[Epoch test #{0}] mca_gen={1:.5f} mca_cov={2:.5f} acc_cov={3:5f}\n"
                        .format(epoch, mca_gen, mca_cov, *loss_checker.mean(0)))

        
        if mca_cov > test_mca_best:
            test_mca_best = mca_cov
            save_model(net,optimizer,C.checkpoints_dir,suff='best')


def get_split_class(source_class_path, target_class_path):

    sourse_classes = open(source_class_path, 'r')
    sourse_classes_list = []
    target_classes = open(target_class_path, 'r')
    target_classes_list = []
    for s_class in sourse_classes:
        s_class = s_class.split()[0]
        sourse_classes_list.append(s_class)
    for t_class in target_classes:
        t_class = t_class.split()[0]
        target_classes_list.append(t_class)
    return sourse_classes_list , target_classes_list


def get_attr_set(source_classes_list, target_classes_list, data_info):
    attr_set = []
    is_test_class = []
    for s_class in source_classes_list:
        attr = data_info[s_class]['attribute']
        attr_set.append(attr)
        is_test_class.append(0)
    for t_class in target_classes_list:
        attr = data_info[t_class]['attribute']
        attr_set.append(attr)
        is_test_class.append(1)
    attr_set = torch.Tensor(attr_set)
    is_test_class = torch.Tensor(is_test_class)
    return attr_set, is_test_class

def get_data_info(data_info_path):
    with open(data_info_path, 'r+') as file:
        content = file.read()
    data_info = json.loads(content)
    return data_info


def main(args):
    set_random_seed(C.seed)  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    map_loc = None if torch.cuda.is_available() else 'cpu'
    source_classes_list , target_classes_list = get_split_class(args.train_class_path, args.test_class_path)
    data_info = get_data_info(args.data_path)
    attr_set, is_test_class = get_attr_set(source_classes_list, target_classes_list, data_info)
    train_iter, test_iter = build_loader(args.img_path, source_classes_list, target_classes_list, data_info, args.batch_size, args.num_workers)
    model = QFSLnet(attr_set.to(device), args.img_encoder_name, args.dataset_name)
    no_grad = [
    'weightnet.weight',
    'weightnet.bias',
    ]
    for name, value in model.named_parameters():
#         print(name)
        if name in no_grad:
            value.requires_grad = False
        else:
            value.requires_grad = True
#     ipdb.set_trace()
    fine_net = [
    'VSBSnet.net.1.weight',
    'VSBSnet.net.1.bias',
    ]
    params_1x = [param for name, param in model.named_parameters()
             if name not in fine_net]
    
    optimizer = torch.optim.SGD([{'params': params_1x},
                                   {'params': model.VSBSnet.net.parameters(),
                                    'lr': args.learning_rate}],
                                lr=args.learning_rate/10,
                                momentum=0.9,
                                weight_decay=C.Model.weights_decay)
    if args.resume and os.path.exists(os.path.join(C.checkpoints_dir, 'model.ckpt')):
        print("load_pretrained_model")
        model.load_state_dict(torch.load(os.path.join(C.checkpoints_dir, 'model.ckpt'),map_location=map_loc))

    train_model(args,model,train_iter,test_iter,QFSLloss,optimizer,is_test_class,device)


if __name__ == "__main__":
    main(get_parser())
