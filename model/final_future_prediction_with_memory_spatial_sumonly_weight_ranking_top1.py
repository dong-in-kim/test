import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory_final_spatial_sumonly_weight_ranking_top1 import *
import torchvision


class Encoder(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel =3):
        super(Encoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
        
        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )

        def chunk_(input, number, dim):
            x_1, x_2, x_3, x_4 = torch.chunk(input, number, dim)
            return x_1, x_2, x_3, x_4

        def stack_(input1, input2, input3, input4, dim):
             return torch.stack([input1, input2, input3, input4], dim)


        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 1, 1, 1),
            torch.nn.ReLU(inplace=False),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=False),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=32, stride=1, padding=0),
            torch.nn.ReLU(inplace=False),
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 3, kernel_size=2, stride=1, padding=0),
            torch.nn.ReLU(inplace=False),
        )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        self.pointwise1 = conv_dw(4, 3, 1)
        self.moduleConv1 = Basic(n_channel*(t_length-1), 64)#3x4,64
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)



    def forward(self, x):

        chunk1a, chunk1b, chunk1c, chunk1d = torch.chunk(x, 4, 1)  # (3x256x256) each four
        chunk1a1 = self.layer1(chunk1a)  # 1x256x256
        chunk1b1 = self.layer1(chunk1b)
        chunk1c1 = self.layer1(chunk1c)
        chunk1d1 = self.layer1(chunk1d)

        conv1 = torch.cat([chunk1a1, chunk1b1, chunk1c1, chunk1d1], 1)  # 4x4x256x256
        pool1 = self.modulePool1(conv1)  # 4x4x128x128
        chunk2a, chunk2b, chunk2c, chunk2d = torch.chunk(pool1, 4, 1)  # 1x128x128 each four

        chunk2a1 = self.layer2(chunk2a)  # 1x128x128
        chunk2b1 = self.layer2(chunk2b)
        chunk2c1 = self.layer2(chunk2c)
        chunk2d1 = self.layer2(chunk2d)
        conv2 = torch.cat([chunk2a1, chunk2b1, chunk2c1, chunk2d1], 1)#4x128x128
        pool2 = self.modulePool1(conv2)  # 4x4x64x64

        chunk3a, chunk3b, chunk3c, chunk3d = torch.chunk(pool2, 4, 1)#1x64x64
        chunk3a1 = self.layer2(chunk3a)  # 1x128x128
        chunk3b1 = self.layer2(chunk3b)
        chunk3c1 = self.layer2(chunk3c)
        chunk3d1 = self.layer2(chunk3d)
        conv3 = torch.cat([chunk3a1, chunk3b1, chunk3c1, chunk3d1], 1)#4x64x64
        pool3 = self.modulePool1(conv3)#4x32x32

        chunk4a, chunk4b, chunk4c, chunk4d = torch.chunk(pool3, 4, 1)#1*32*32
        chunk5 = self.pointwise1(pool3)#3*32*32

        chunk4a1 = self.layer3(chunk4a)#1 64x32x32
        chunk4b2 = self.layer4(chunk4b)#2 64x1x1 expand
        chunk4b2_= chunk4b2.repeat(32, 32)

        #4*12*256*256
        tensorConv1 = self.moduleConv1(x)#4*64*256*256
        tensorPool1 = self.modulePool1(tensorConv1)#4*64*128*128
        #print(np.shape(tensorConv1))
        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)
        #print(np.shape(tensorConv2))
        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        #print(np.shape(tensorConv3))
        #print(np.shape(tensorConv4))
        return tensorConv4, tensorConv1, tensorConv2, tensorConv3

    
    
class Decoder(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel =3):
        super(Decoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
                
        
        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
        
        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels = nc, out_channels=intOutput, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
      
        self.moduleConv = Basic(1024, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(128,n_channel,64)
        self.fc1 = nn.Linear(3*256*256,1200)
        self.fc2 = nn.Linear(1200,100)
        self.fc3 = nn.Linear(100,4)
        
        
    def forward(self, x, skip1, skip2, skip3):
        
        tensorConv = self.moduleConv(x)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        cat4 = torch.cat((skip3, tensorUpsample4), dim = 1)
        
        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim = 1)
        
        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim = 1)
        
        output = self.moduleDeconv1(cat2)
        ##
        tensordirection1 = torch.flatten(output,1)
        tensordirection2 = F.relu(self.fc1(tensordirection1))
        tensordircetion3 = F.relu(self.fc2(tensordirection2))
        direction_out = self.fc3(tensordircetion3)


                
        return output, direction_out
    


class convAE(torch.nn.Module):
    def __init__(self, n_channel =3,  t_length = 5, memory_size = 10, feature_dim = 512, key_dim = 512, temp_update = 0.1, temp_gather=0.1):
        super(convAE, self).__init__()

        self.encoder = Encoder(t_length, n_channel)
        self.decoder = Decoder(t_length, n_channel)
        self.memory = Memory(memory_size,feature_dim, key_dim, temp_update, temp_gather)
       

    def forward(self, x, keys,train=True):

        fea, skip1, skip2, skip3 = self.encoder(x)
        if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = self.memory(fea, keys, train)
            output, direction_out = self.decoder(updated_fea, skip1, skip2, skip3)

            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss, direction_out
        
        #test
        else:
            updated_fea, keys, softmax_score_query, softmax_score_memory,query, top1_keys, keys_ind, compactness_loss = self.memory(fea, keys, train)
            output, direction_out = self.decoder(updated_fea, skip1, skip2, skip3)
            
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss, direction_out
        
                                          



    
    
