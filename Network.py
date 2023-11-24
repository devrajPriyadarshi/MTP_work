from random import sample

import numpy as np

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, AdaptiveAvgPool1d, Flatten
import torchvision.models as Models

from FoldingNet import Decoder

# from Vizualization import ComparePointClouds

# from torchinfo import summary

# if torch.cuda.is_available():
#     device = torch.device("cuda:1")
#     # device = torch.device("cpu")
#     torch.cuda.set_device(device)
# else:
#     device = torch.device("cpu")

class FeatureExtractor(nn.Module):
    """ Input Size =  [BATCH x NUM_VIEWS x C x H x W]
        Output Size = [BATCH x NUM_VIEWS x d_feature] """
    def __init__(self, num_views):
        super(FeatureExtractor, self).__init__()
        
        self.NUM_VIEWS = num_views

        # VGG16_weights = Models.VGG16_Weights.IMAGENET1K_V1
        # self.VGG16_model = Models.vgg16(weights=VGG16_weights).to(device)
        self.VGG16_model = Models.vgg16(pretrained=True)

        for param in self.VGG16_model.parameters():
            param.requires_grad = False

    def forward(self, inp):
        """ Input is a  [BATCH x NUM_VIEWS x C x H x W]  vector
            Model takes [BATCH x C x H x W]              vector """
        out = torch.stack( [ self.VGG16_model(t) for t in inp])
        return out

class ImageEncoder(nn.Module):
    def __init__(self, num_views, num_heads, num_layer):
        super(ImageEncoder, self).__init__()

        self.IMAGE_FEATURE = 1000
        self.NUM_VIEWS = num_views
        self.NUM_HEADS = num_heads
        self.NUM_LAYER = num_layer
        BATCH_FIRST = True

        self.FeatureExtractor = FeatureExtractor(num_views=self.NUM_VIEWS)
        self.TransformerEncoderLayer = TransformerEncoderLayer(d_model=self.IMAGE_FEATURE, nhead=self.NUM_HEADS, batch_first=BATCH_FIRST)
        self.TransformerEncoder = TransformerEncoder(self.TransformerEncoderLayer, num_layers=self.NUM_LAYER)
        self.AdaptiveAvgPool = AdaptiveAvgPool1d(output_size=1)
        self.Flatten = Flatten()

    def forward(self, x):
        x = self.FeatureExtractor(x)
        x = self.TransformerEncoder(x)
        x = torch.transpose(x, 1, 2)
        x = self.AdaptiveAvgPool(x)
        x = self.Flatten(x)

        return x

    # def printSummary(self):
    #     # summary(self.FeatureExtractor, input_size=(1, self.NUM_VIEWS, 3, 224, 224))
    #     # summary(self.TransformerEncoderLayer, input_size=(1,self.NUM_VIEWS,self.IMAGE_FEATURE))
    #     # summary(self.TransformerEncoder     , input_size=(1,self.NUM_VIEWS,self.IMAGE_FEATURE))


class FoldingDecoder(nn.Module):
    def __init__(self, in_channel = 1000):
        super(FoldingDecoder, self).__init__()
        self.Decoder = Decoder(in_channel=in_channel)

    def forward(self, x):
        x = self.Decoder(x)
        return x

class Network(nn.Module):
    def __init__(self, num_views, num_heads, num_layer):
        super(Network, self).__init__()
        
        self.encoder = ImageEncoder(num_views=num_views, num_heads=num_heads, num_layer=num_layer)
        self.decoder = FoldingDecoder(in_channel=1000)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.transpose(x, 1, 2)
        return x
    
# if __name__ == '__main__':

#     BATCH_SIZE = 1
#     NUM_VIEWS = 24
#     NUM_HEADS = 1
#     NUM_LAYER = 2

#     # Take a input sample
#     pc_Tensor = fixPointcloudOrientation(torch.Tensor(np.load("tests/chair_sample/pointcloud_1024.npy")))
#     img_Tensor = torch.Tensor(np.array([cv2ToTensor(cv2.imread("tests/chair_sample/rendering/"+str(idx//10)+str(idx%10)+".png")) for idx in sample(range(24), NUM_VIEWS) ]))

#     # encoder = ImageEncoder(num_views=NUM_VIEWS, num_heads=NUM_HEADS, num_layer=NUM_LAYER)
#     # summary(encoder, input_size=[BATCH_SIZE, NUM_VIEWS, 3, 224, 224], row_settings=('depth', 'hide_recursive_layers'))
#     # # encoder.printSummary()
    
#     # decoder = FoldingDecoder()
#     # summary(decoder, input_size=[BATCH_SIZE, 1000], row_settings=('depth', 'hide_recursive_layers'))
    
#     net = Network(num_views=NUM_VIEWS, num_heads=NUM_HEADS, num_layer=NUM_LAYER).to(device)
#     inp = img_Tensor.unsqueeze(0).to(device)
#     gt_pc = pc_Tensor.to(device)
#     res_pc = net(inp)

#     plot = ComparePointClouds(gt_pc, res_pc.squeeze(0).permute(1,0))
#     plot.show()