## Transductive Unbiased Embedding for Zero-Shot Learning

Code supporting the paper:

*Jie Song, Chengchao Shen, Yezhou Yang, Yang Liu, Mingli Song.
[Transductive Unbiased Embedding for Zero-Shot Learning. ](https://openaccess.thecvf.com/content_cvpr_2018/papers/Song_Transductive_Unbiased_Embedding_CVPR_2018_paper.pdf)
CVPR 2018*


If you find this code useful in your research, please consider citing using the
following BibTeX entry:

```
@InProceedings{Jie CVPR2018,
author = {Jie Song, Chengchao Shen, Yezhou Yang, Yang Liu, Mingli Song},
title = {Transductive Unbiased Embedding for Zero-Shot Learning},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2018}
}
```

### Installation

This code uses Python 3.8 and PyTorch 1.9.0 cuda version 10.2.

- Installing PyTorch:
```bash
$ conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
```

- Install dependencies
```bash
$ pip install -r requirements.txt
```

### Tour
#### code

1.```ims2file.py```: Script for preparing LMDBs used in ```train.py``` and ```sample.py```.

2.```train.py```: Training script.

3.```sample.py```: Evaluation script.

4.```QFSLnet.py```: Defines the QFSLmodel.

#### data

1.Partition into source and target classes: ```classes.txt```,```trainvalclasses.txt```,```testclasses.txt``` .

2.class attributes names: ```attributes.txt```

3.class attributes labels: ```class_attribute_labels_continuous.txt```


### Dataset

- Download [AWA2](https://cvml.ist.ac.at/AwA2/AwA2-data.zip) 
- Download [CUB](https://data.caltech.edu/tindfiles/serve/1239ea37-e132-42ee-8c09-c383bb54e7ff/) 
- Download [SUN](https://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB_Images.tar.gz) 



### Images to Image_LMDB and Information_json

Prepare for training

```bash
$ python ims2file.py --dataset_name AWA2/CUB/SUN --class_names_path path_to_dataset's_classes_names \
--dataset_img_path path_to_dataset's_Imagedir --dataset_attr_path path_to_allclasses's_attributes \
--save_path path_for_LMDB_to_save
```



### Training


We train our model in this way:


```bash
python train.py --dataset_name AWA2 --img_encoder_name AlexNet/ResNet101/VGG19/GoogLeNet \
--train_class_path path_to_source_class --test_class_path path_to_target_class \ 
--data_path path_to_Information_json --img_path path_to_lmdb --learning_rate 0.005 \ 
--bias_weight 0.2  --num_epochs 5000 --batch_size 64 --num_workers 4   
```

Example

```bash
python train.py --dataset_name AWA2 --img_encoder_name AlexNet \ 
--train_class_path ../data/AWA2/standard_split/trainvalclasses.txt \ 
--test_class_path ../data/AWA2/standard_split/testclasses.txt \ 
--data_path ../data_save/AWA2/data_info.json \ 
--img_path ../data_save/AWA2/lmdb \ 
--learning_rate 0.005 --bias_weight 0.2 --num_epochs 5000 --batch_size 64 --num_workers 4 
```

Check training progress in ```src/checker/logger```:
Model save in ```src/checker/checkpoints```:


### Evaluation
Example

```bash
python sample.py --dataset_name AWA2 --img_encoder_name AlexNet \
 --train_class_path ../data/AWA2/standard_split/trainvalclasses.txt \
 --test_class_path ../data/AWA2/standard_split/testclasses.txt  \
  --data_path ../data_save/AWA2/data_info.json \ 
 --img_path ../data_save/AWA2/lmdb  --batch_size 64 --num_workers 4
```

- This script will return  Mean class accuracy for target classes in conventional setting and generalized setting


