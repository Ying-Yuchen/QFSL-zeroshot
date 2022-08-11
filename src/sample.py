import torch
import torchvision
import sys
import os
from utils.config import Config as C
from torch.nn import functional as F
from torch import nn
from utils.utils import set_random_seed, LossChecker, test_acc
from models.QFSLnet import QFSLnet
from data_loader import Corpus
from tqdm import tqdm
import ipdb
from collections import Counter
from args import get_parser
import json



def build_loader(img_path, source_classes_list, target_classes_list, data_info, batch_size,num_wokers):
    corpus = Corpus(img_path, source_classes_list, target_classes_list, data_info, batch_size,num_wokers)
    return corpus.test_dataloader
    


def get_eval(args,model,test_iter,is_test_class,device):
    net = model.to(device)
    loss_checker = LossChecker(1)
    counter_gen = Counter()
    counter_cov = Counter()
    counter_true = Counter()
    test_pgbar = tqdm(test_iter)
    mca_gen, mca_cov = 0,0
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
            "[test#{0}]lr={4:5f} mca_gen={1:.5f} mca_cov={2:.5f} acc_cov={3:5f}"
                .format(0, mca_gen, mca_cov, *loss_checker.mean(0),args.learning_rate)
        )
        
    with open(C.test_score_path, "a") as fout:
            fout.write( "[test #{0}] mca_gen={1:.5f} mca_cov={2:.5f} acc_cov={3:5f}\n"
                        .format(0, mca_gen, mca_cov, *loss_checker.mean(0)))

        


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
    is_test_attr = []
    for s_class in source_classes_list:
        attr = data_info[s_class]['attribute']
        attr_set.append(attr)
        is_test_attr.append(0)
    for t_class in target_classes_list:
        attr = data_info[t_class]['attribute']
        attr_set.append(attr)
        is_test_attr.append(1)
    attr_set = torch.Tensor(attr_set)
    is_test_attr = torch.Tensor(is_test_attr)
    return attr_set, is_test_attr

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
    test_iter = build_loader(args.img_path, source_classes_list, target_classes_list, data_info, args.batch_size, args.num_workers)
    model = QFSLnet(attr_set.to(device), args.img_encoder_name, args.dataset_name)
    
    if  os.path.exists(os.path.join(C.checkpoints_dir, 'model.ckpt')):
        print("load_pretrained_model")
        model.load_state_dict(torch.load(os.path.join(C.checkpoints_dir, 'model.ckpt'),map_location=map_loc))

    get_eval(args,model,test_iter,is_test_class,device)


if __name__ == "__main__":
    main(get_parser())