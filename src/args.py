import argparse
import os


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='AWA2',
                        help='AWA2 / CUB / SUN')

    parser.add_argument('--img_encoder_name', type=str, default='AlexNet',
                        help="img_encoder model. Options: 'AlexNet', 'GoogLeNet', 'VGG19' , 'ResNet101' , 'SwinTransformer'")

    parser.add_argument('--train_class_path', type=str, default='../data/AWA2/standard_split/trainvalclasses.txt',
                        help='path of train class names')

    parser.add_argument('--test_class_path', type=str, default='../data/AWA2/standard_split/testclasses.txt',
                        help='path of train class names')

    parser.add_argument('--data_path', type=str, default='../data_save/AWA2/data_info.json',
                        help='path of json file which has saved image names and class attributes')

    parser.add_argument('--img_path', type=str, default='../data_save/AWA2/lmdb',
                        help='path of lmdb file of images')


    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='base learning rate')

    parser.add_argument('--bias_weight', type=float, default=0.2,
                        help='the weight of bias loss')

    parser.add_argument('--resume_epoch_nums', type=int, default=0,
                        help='if resume model, define the start epoch num')

    parser.add_argument('--num_epochs', type=int, default=400,
                        help='maximum number of epochs')

    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='resume training from the checkpoint in model_name')
    parser.set_defaults(resume=False)

    args = parser.parse_args()
    assert args.dataset_name in ("AWA2","CUB","SUN")
    assert args.img_encoder_name in ('AlexNet', 'GoogLeNet', 'VGG19' , 'ResNet101')

    return args