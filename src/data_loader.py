import torch
import torchvision
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
import sys
import os
import lmdb
import ipdb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils.config import Config as C
from torch.nn import functional as F
import numpy as np
import json



class QFSLdataset(Dataset):
    def __init__(self, img_path, source_classes_list, target_classes_list, data_info, split):
        self.split = split
        self.source_classes_list = source_classes_list
        self.target_classes_list = target_classes_list
        self.data_info = data_info
        self.data = []
        self.image_file = lmdb.open(img_path, max_readers=1, readonly=True,
                                    lock=False, readahead=False, meminit=False)

        if self.split == 'train':
            class_num = 0
            is_train = 1
            for s_class in self.source_classes_list:
                class_attr = torch.Tensor(data_info[s_class]['attribute'])
                for pic_name in data_info[s_class]['pictures']:
                    self.data.append((pic_name, class_num, is_train, class_attr))
                class_num += 1

            class_num = len(self.source_classes_list)
            is_train = 0
            for t_class in self.target_classes_list:
                class_attr = torch.Tensor(data_info[t_class]['attribute'])
                for pic_name in data_info[t_class]['pictures']:
                    self.data.append((pic_name, class_num, is_train, class_attr))
                class_num += 1

        else :
            class_num = len(self.source_classes_list)
            is_train = 0
            for t_class in self.target_classes_list:
                class_attr = torch.Tensor(data_info[t_class]['attribute'])
                for pic_name in data_info[t_class]['pictures']:
                    self.data.append((pic_name, class_num, is_train, class_attr))
                class_num += 1


        image_size = C.Model.img_size

        self.transform_test = transforms.Compose([
#             transforms.Resize((image_size, image_size)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
#             transforms.Resize((image_size, image_size)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx]
        pic_name, class_num, is_train, class_attr = pos
        with self.image_file.begin(write=False) as txn:
            image = txn.get(pic_name.encode())
            image = np.fromstring(image, dtype=np.uint8)
            image = np.reshape(image, (256, 256, 3))
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        if self.split=='train':
            image = self.transform_train(image)
        else :
            image = self.transform_test(image)

        return (image, class_num, is_train, class_attr)


class Corpus(object):

    def __init__(self, img_path, source_classes_list, target_classes_list, data_info, batch_size,num_workers,is_eval=False, dataset_cls=QFSLdataset):
        self.obj2one_hot = None
        self.dataset_cls = dataset_cls
        self.is_eval = is_eval
        self.train_dataset = None
        self.train_dataloader = None
        self.test_dataset = None
        self.test_dataloader = None

        self.build(img_path, source_classes_list, target_classes_list, data_info, batch_size,num_workers)

    def __len__(self):
        return len(self.train_dataset)

    def build(self, img_path, source_classes_list, target_classes_list, data_info, batch_size,num_workers):
        if not self.is_eval:
            self.train_dataset = self.build_dataset(
                img_path,
                source_classes_list,
                target_classes_list,
                data_info,
                'train'
            )
            self.train_dataloader = self.build_dataloader(self.train_dataset, batch_size,num_workers)

        self.test_dataset = self.build_dataset(
            img_path,
            source_classes_list,
            target_classes_list,
            data_info,
            'test'
        )
        
        self.test_dataloader = self.build_dataloader(self.test_dataset, batch_size,num_workers)
    

    def build_dataset(self,img_path, source_classes_list, target_classes_list, data_info, split):
        dataset = self.dataset_cls(
            img_path,
            source_classes_list,
            target_classes_list,
            data_info,
            split
        )
        return dataset

    def build_dataloader(self,dataset, batch_size,num_workers):
        data_loader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers,
        )
        return data_loader

