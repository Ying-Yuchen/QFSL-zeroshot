
from tqdm import tqdm
import numpy as np
import argparse
import lmdb
import torch
import json
from torchvision import transforms
from PIL import Image
import sys
import os
import ipdb
from torch.nn import functional as F
# import skimage.transform as skt
from collections import OrderedDict

MAX_SIZE = 8e9

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='AWA2',
                        help='AWA2 / SUN / CUB')
    parser.add_argument('--class_names_path', type=str, default='../data/AWA2/classes.txt',
                        help='the text which record all classes names')
    parser.add_argument('--attr_names_path', type=str, default='../data/AWA2/attributes.txt',
                        help='the text which record all attributes names')
    parser.add_argument('--dataset_img_path', type=str, default='../AWA2dataset/JPEGImages',
                        help='the folder containing pictures of all classes')
    parser.add_argument('--dataset_attr_path', type=str, default='../data/AWA2/class_attribute_labels_continuous.txt',
                        help='the text which record all classes attributes labels')
    parser.add_argument('--save_path', type=str, default='../data_save/',
                        help='folder to save lmdb files')


    args = parser.parse_args(sys.argv[1:])
    assert args.dataset_name in ('AWA2', 'CUB', 'SUN')
    return args

def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def load_and_resize(file_path, imscale):

    transf_list = []
    transf_list.append(transforms.Resize(imscale))
    transf_list.append(transforms.CenterCrop(imscale))
    transform = transforms.Compose(transf_list)

    img = Image.open(file_path).convert('RGB')
    img = transform(img)

    return img

#Gets a normalized set of attribute vectors
def get_att(codes_fn, semantics_fn, mean_correction=True):

    semantics = [line.strip().split()[1] for line in open(semantics_fn)]
    codes = np.loadtxt(codes_fn).astype(float)
    if codes.max() > 1:
        codes /= 100.

    # Set undefined codes (typically marked with -1) to the mean
    code_mean = codes[: , :].mean(axis=0)
    for s in range(len(semantics)):
        codes[codes[:, s] < 0, s] = code_mean[s] if mean_correction else 0.5

    # Mean correction
    if mean_correction:
        for s in range(len(semantics)):
            codes[:, s] = codes[:, s] - code_mean[s] + 0.5

    constrains = OrderedDict([(sem, {'codewords': np.array([[-1, 1]]),
                                     'idxDim': s,
                                     'labels': None
                                     })
                              for s, sem in enumerate(semantics)])
    for s, sem in enumerate(semantics):
        src_lbl = np.zeros((2, len(codes)))
        src_lbl[0, :] = 1 - codes[: , s]
        src_lbl[1, :] = codes[:, s]
        constrains[sem]['labels'] = src_lbl

    sem_labels = [const['labels'] for const in constrains.values()]
    sem_labels = np.concatenate([np.expand_dims(s.argmax(axis=0), 1) for s in sem_labels], 1)
    sem_labels = sem_labels[:, :]

    code_dim = [const['codewords'].shape[0] for const in constrains.values()]
    num_states = [const['codewords'].shape[1] for const in constrains.values()]
    constrain_cw = [const['codewords'] for const in constrains.values()]
    codes, c1, c2 = np.zeros((sum(code_dim), sum(num_states))), 0, 0
    for cw in constrain_cw:
        codes[c1:c1 + cw.shape[0], c2:c2 + cw.shape[1]], c1, c2 = cw, c1 + cw.shape[0], c2 + cw.shape[1]
    selector = np.concatenate([const['labels'] for const in constrains.values()], axis=0)

    cw = np.dot(codes, selector)
    cw = cw / ((cw ** 2).sum(axis=0, keepdims=True)) ** 0.5

    return sem_labels.tolist(), cw.T.tolist()

#get image lmdb and dataInfo
def main(args):
    cap_dict = {'AWA2': 9e9 ,'CUB': 3e9 , 'SUN': 5e9}
    MAX_SIZE = cap_dict[args.dataset_name]
    save_path = os.path.join(args.save_path,args.dataset_name)
    make_dir(save_path)
    file = lmdb.open(os.path.join(save_path, 'lmdb'), map_size=int(MAX_SIZE))
    with file.begin() as txn:
        present_entries = [key for key, _ in txn.cursor()]
    class2picname_and_attri = {}
    classes_names = open(args.class_names_path, 'r')
    sem_labels, codewords = get_att(codes_fn=args.dataset_attr_path,semantics_fn=args.attr_names_path)
    for id, class_ in enumerate(classes_names):
        print(id)
        print(class_)
        class_name = class_.split()[1]
        codeword = codewords[id]
        sem_label = sem_labels[id]
        class2picname_and_attri[class_name] = {'pictures':[],
                                               'attribute':codeword,
                                               'sem_label':sem_label ,
                                               'class_num':id,
                                               }
        if args.dataset_name == 'SUN':
            file_dir = os.path.join(args.dataset_img_path, class_name[0], class_name)
            if not os.path.exists(file_dir):
                class_name_split = class_name.split('_')
                class_name_first = '_'.join(class_name_split[0:-1])
                class_name_last = class_name_split[-1]
                file_dir1 = os.path.join(args.dataset_img_path, class_name[0], class_name_first, class_name_last)
                class_name_first = class_name_split[0]
                class_name_last = '_'.join(class_name_split[1:])
                file_dir2 = os.path.join(args.dataset_img_path, class_name[0], class_name_first, class_name_last)
                class_name_first = '_'.join(class_name_split[0:2])
                class_name_last = '_'.join(class_name_split[2:])
                file_dir3 = os.path.join(args.dataset_img_path, class_name[0], class_name_first, class_name_last)
                if  os.path.exists(file_dir1):
                    file_dir = file_dir1
                if  os.path.exists(file_dir2):
                    file_dir = file_dir2
                if  os.path.exists(file_dir3):
                    file_dir = file_dir3
        else :
            file_dir = os.path.join(args.dataset_img_path, class_name)

        files_name = getFlist(file_dir)
        for file_name in tqdm(files_name):
            class2picname_and_attri[class_name]['pictures'].append(file_name)
            file_path = os.path.join(file_dir, file_name)
            if file_name.encode() not in present_entries:
                im = load_and_resize(file_path,256)
                im = np.array(im).astype(np.uint8)
                with file.begin(write=True) as txn:
                    txn.put(file_name.encode(), im)
    data_json = json.dumps(class2picname_and_attri)
    json_file_path = os.path.join(save_path, 'data_info.json')
    with open(json_file_path, 'w+') as file:
        file.write(data_json)




def getFlist(file_dir):
    for root, dirs, files in os.walk(file_dir):

        return files

def test():

    image_file = lmdb.open(os.path.join('../data_save/AWA2', 'lmdb'), max_readers=1, readonly=True,
                                    lock=False, readahead=False, meminit=False)
    with image_file.begin(write=False) as txn:
        for key, value in txn.cursor(): 
            image = txn.get(key)
            image = np.frombuffer(image, dtype=np.uint8)
            image = np.reshape(image, (256, 256, 3))
            image = Image.fromarray(image.astype('uint8'), 'RGB')
            image.show()



if __name__ == "__main__":
    args = get_parse()
    main(args)

    # test()
