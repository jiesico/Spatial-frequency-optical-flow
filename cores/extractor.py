import torch
import torch.nn as nn
import torch.nn.functional as F
from mmtm import MMTM
import copy
import numpy as np
import math
from copy import deepcopy
from setrans import SETransConfig, SelfAttVisPosTrans
from FourierTransformer import DCTAttention
from FFTformer import Fuse,Fusecross
def get_emb(sin_inp):
#     """
#     Gets a base embedding for one dimension with sin and cos intertwined
#     """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)
def pos_emd2d(tensor, dim=2):

    bs, _, height, width = tensor.shape
    device = tensor.device
    pe = torch.zeros(dim, height, width).to(device)#############
    dim = int(dim / 2)
    div_term = torch.exp(-torch.arange(0., dim, 2) * (math.log(10000.0) / dim)).to(device)
    pos_w = torch.arange(0., width).unsqueeze(1).to(device)############
    pos_h = torch.arange(0., height).unsqueeze(1).to(device)############
    pe[0:dim:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1).to(device)
    pe[1:dim:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1).to(device)
    pe[dim::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width).to(device)
    pe[dim + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width).to(device)
    pe = pe.unsqueeze(0).repeat(bs, 1, 1, 1)
    return pe
def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        # self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)
#
    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))
#
#
class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names
#
    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class CONV(nn.Module):
    def __init__(self, in_dim=1, norm_fn='batch'):
        super(CONV, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            b_norm = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            b_norm = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            b_norm = nn.InstanceNorm2d(64)
        else:
            b_norm = nn.Sequential()
        self.relu1 = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3),
            b_norm,
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        self.conv_rgb = CONV(in_dim=3, norm_fn=norm_fn)
        self.conv_dep = CONV(in_dim=1, norm_fn=norm_fn)
        # self.rgbpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        # self.depthpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.in_planes = 64
        layer1 = self._make_layer(64, stride=1)
        layer2 = self._make_layer(96, stride=2)
        layer3 = self._make_layer(128, stride=2)
        # self.mmtmrgb =

        self.layerrgb1 = nn.Sequential(layer1, layer2)#, layer3)
        self.layerdepth1 = copy.deepcopy(nn.Sequential(layer1, layer2))
        self.layerrgb1 = layer1#, layer3)
        self.layerdepth1 = copy.deepcopy(layer1)
        self.layerrgb2 = layer2
        self.layerdepth2 = copy.deepcopy(layer2)
        self.layerrgb3 = layer3
        self.layerdepth3 = copy.deepcopy(layer3)
        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        self.conv2d = nn.Conv2d(128, output_dim, kernel_size=1)
        # self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)
        self.dropout = None
        self.fuse1 = Fuse(96,4)
        self.fuse2 = Fuse(96,4)
        self.fuse3 = Fusecross(96,4)

        # self.fuse4 = Fuse(128,2)
        # self.fuse5 = Fuse(128,2)
        # self.fuse6 = Fusecross(128,2)
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def _make_layers(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):


        a = x[0]
        b = x[1]


        # pe = pos_emd2d(b)
        # b = torch.cat([b,pe],1)
        # print('2')
        # print(a.shape,b.shape)

        a = self.conv_rgb(a)
        b = self.conv_dep(b)
        # print('before-sa',a.shape)
#######################################################
        # apool = self.rgbpool1(a)
        # bpool = self.depthpool1(b)
        # apool = self.fuse1(apool,apool)
        # bpool = self.fuse2(bpool,bpool)
        # apool,bpool = self.fuse3(apool,bpool)
        # a = F.interpolate(apool, size=(a.shape[2], a.shape[3]), mode='bilinear', align_corners=False)
        # b = F.interpolate(bpool, size=(a.shape[2], b.shape[3]), mode='bilinear', align_corners=False)
#############################################################
#######################################################

#############################################################
        a = self.layerrgb1(a)
        b = self.layerdepth1(b)
        # print('after-sa1',a.shape)
        a = self.layerrgb2(a)
        b = self.layerdepth2(b)
        # print('after-sa2',a.shape)
        a = self.fuse1(a,a)
        b = self.fuse2(b,b)
        a,b = self.fuse3(a,b)
        a = self.layerrgb3(a)
        b = self.layerdepth3(b)
        # apool = self.fuse4(a,a)
        # bpool = self.fuse5(b,b)
        # a,b = self.fuse6(apool,bpool)
        a = self.conv2(a)
        b = self.conv2d(b)
        # x = torch.cat([a, b], 1)
        # x = self.conv3(x)

        return a,b


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        print('3')
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.mmtm0(x)
        x = self.layer2(x)
        x = self.mmtm1(x)
        x = self.layer3(x)
        x = self.mmtm2(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
