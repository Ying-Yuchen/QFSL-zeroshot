import torch
import torch.nn as nn
import ipdb
from .encoder_models.resnet import resnet101
from .encoder_models.AlexNet import alexnet
from .encoder_models.GoogLenet import googlenet
from .encoder_models.VGG import VGG19




class EncoderCNN(nn.Module):
    def __init__(self, img_encoder='ResNet101', pretrained=True):
        """Load the pretrained model witch has been replaced top fc layer."""
        super(EncoderCNN, self).__init__()
        self.img_encoder = img_encoder
        self.net = None
        self.output_size = None
        if self.img_encoder=='AlexNet':
            alex = alexnet(pretrained=True)
            self.net = alex
            self.output_size = 4096
        if self.img_encoder=='ResNet101':
            res101 = resnet101(pretrained=True)
            self.net = res101
            self.output_size = res101.fc.in_features
        if self.img_encoder=='VGG19':
            vgg = VGG19(pretrained=True)
            self.net = vgg
            self.output_size = 4096
        if self.img_encoder=='GoogLeNet':
            google = googlenet(pretrained=True, progress=True)
            self.net = google
            self.output_size = 1024


    def forward(self, images, keep_cnn_gradients=False):
        """Extract feature vectors from input images."""
        if keep_cnn_gradients:
            raw_conv_feats = self.net(images)
        else:
            with torch.no_grad():
                raw_conv_feats = self.net(images)
        features = raw_conv_feats
        # ipdb.set_trace()
        return features

class VSBSnet(nn.Module):
    def __init__(self, img_encoder='ResNet101',  input_size=1024, output_size=85):
        super(VSBSnet, self).__init__()
        self.img_encoder = img_encoder
        self.net =  nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(input_size,output_size),
                )
        
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)

        self.net.apply(init_weights)

    def forward(self, img_features):
#         ipdb.set_trace()
        semantic_embedding = self.net(img_features)
        return semantic_embedding


