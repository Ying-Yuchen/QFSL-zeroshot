import torch
from torch import nn
import ipdb
import torchvision
import sys
import os
from models.Encoder import EncoderCNN, VSBSnet
from torch.nn import functional as F




class QFSLnet(nn.Module):
    def __init__(self , attr_set, img_encoder_name, dataset_name):
        super(QFSLnet, self).__init__()
        dataset_info = {'AWA2':{'attr_dim':85,'class_num':50},
                             'CUB':{'attr_dim':312,'class_num':200},
                             'SUN':{'attr_dim':102,'class_num':717}}

        attr_dim = dataset_info[dataset_name]['attr_dim']
        class_num = dataset_info[dataset_name]['class_num']
        self.attr_set = attr_set
        self.dataset_name = dataset_name
        self.img_encoder = EncoderCNN(img_encoder=img_encoder_name, pretrained=True)
        self.VSBSnet = VSBSnet(img_encoder=img_encoder_name, input_size=self.img_encoder.output_size, output_size=attr_dim)
        self.weightnet = nn.Linear(in_features=attr_dim, out_features= class_num)
        self.weightnet.weight.data = attr_set
        self.weightnet.bias.data = torch.zeros(class_num)
        self.weightnet_target = nn.Linear(in_features=attr_dim, out_features= class_num)



    def forward(self,img_tensor,is_train_net=True):
        if self.dataset_name=='SUN':
            img_fea = self.img_encoder(img_tensor,keep_cnn_gradients=False)
        else:
            img_fea = self.img_encoder(img_tensor,keep_cnn_gradients=True)
        senamic_embedding = self.VSBSnet(img_fea)
        senamic_score = self.weightnet(senamic_embedding)
        pos = torch.softmax(senamic_score,dim=1)
        #ipdb.set_trace()
        return pos
        

